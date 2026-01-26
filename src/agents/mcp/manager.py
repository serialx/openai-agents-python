from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable

from ..logger import logger
from .server import MCPServer


async def _run_with_timeout_in_task(
    func: Callable[[], Awaitable[Any]], timeout_seconds: float | None
) -> None:
    """Run an async function with optional timeout."""
    if timeout_seconds is None:
        await func()
    else:
        await asyncio.wait_for(func(), timeout=timeout_seconds)


class _ServerWorker:
    """Worker that manages a server connection in its own task context.

    This preserves task affinity for libraries like AnyIO that need consistent
    task context between connect and cleanup.
    """

    def __init__(
        self,
        server: MCPServer,
        connect_timeout_seconds: float | None,
        cleanup_timeout_seconds: float | None,
    ) -> None:
        self.server = server
        self.connect_timeout_seconds = connect_timeout_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self._task: asyncio.Task[None] | None = None
        self._connect_event: asyncio.Event = asyncio.Event()
        self._cleanup_event: asyncio.Event = asyncio.Event()
        self._cleanup_done_event: asyncio.Event = asyncio.Event()
        self._connect_error: BaseException | None = None
        self._is_done = False

    @property
    def is_done(self) -> bool:
        return self._is_done

    async def connect(self) -> None:
        """Start the worker task and wait for connection."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await self._connect_event.wait()
        if self._connect_error is not None:
            raise self._connect_error

    async def cleanup(self) -> None:
        """Signal cleanup and wait for completion."""
        self._cleanup_event.set()
        await self._cleanup_done_event.wait()

    async def _run(self) -> None:
        """Main worker loop: connect, wait for cleanup signal, then cleanup."""
        try:
            await _run_with_timeout_in_task(self.server.connect, self.connect_timeout_seconds)
        except BaseException as exc:
            self._connect_error = exc
            self._connect_event.set()
            self._is_done = True
            return

        self._connect_event.set()
        await self._cleanup_event.wait()

        try:
            await _run_with_timeout_in_task(self.server.cleanup, self.cleanup_timeout_seconds)
        finally:
            self._is_done = True
            self._cleanup_done_event.set()


class MCPServerManager(AbstractAsyncContextManager["MCPServerManager"]):
    """Manage MCP server lifecycles and expose only connected servers.

    Use this helper to keep MCP connect/cleanup on the same task and avoid
    run failures when a server is unavailable. The manager will attempt to
    connect each server and then expose the connected subset via
    `active_servers`.

    Basic usage:
        async with MCPServerManager([server_a, server_b]) as manager:
            agent = Agent(
                name="Assistant",
                instructions="...",
                mcp_servers=manager.active_servers,
            )

    FastAPI lifespan example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with MCPServerManager([server_a, server_b]) as manager:
                app.state.mcp_manager = manager
                yield

        app = FastAPI(lifespan=lifespan)

    Important behaviors:
    - `active_servers` only includes servers that connected successfully.
      `failed_servers` holds the failures and `errors` maps servers to errors.
    - `drop_failed_servers=True` removes failed servers from `active_servers`
      (recommended). If False, `active_servers` will still include all servers.
    - `strict=True` raises on the first connection failure. If False, failures
      are recorded and the run can proceed with the remaining servers.
    - `reconnect(failed_only=True)` retries failed servers and refreshes
      `active_servers`.
    - `connect_in_parallel=True` uses a dedicated worker task per server to
      allow concurrent connects while preserving task affinity for cleanup.
    """

    def __init__(
        self,
        servers: Iterable[MCPServer],
        *,
        connect_timeout_seconds: float | None = 10.0,
        cleanup_timeout_seconds: float | None = 10.0,
        drop_failed_servers: bool = True,
        strict: bool = False,
        suppress_cancelled_error: bool = True,
        connect_in_parallel: bool = False,
    ) -> None:
        self._all_servers = list(servers)
        self._active_servers = list(servers)
        self.connect_timeout_seconds = connect_timeout_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.drop_failed_servers = drop_failed_servers
        self.strict = strict
        self.suppress_cancelled_error = suppress_cancelled_error
        self.connect_in_parallel = connect_in_parallel
        self._workers: dict[MCPServer, _ServerWorker] = {}

        self.failed_servers: list[MCPServer] = []
        self._failed_server_set: set[MCPServer] = set()
        self._connected_servers: set[MCPServer] = set()
        self.errors: dict[MCPServer, BaseException] = {}

    @property
    def active_servers(self) -> list[MCPServer]:
        """Return the active MCP servers after connection attempts."""
        return list(self._active_servers)

    @property
    def all_servers(self) -> list[MCPServer]:
        """Return all MCP servers managed by this instance."""
        return list(self._all_servers)

    async def __aenter__(self) -> MCPServerManager:
        await self.connect_all()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        await self.cleanup_all()
        return None

    async def connect_all(self) -> list[MCPServer]:
        """Connect all servers in order and return the active list."""
        previous_connected_servers = set(self._connected_servers)
        previous_active_servers = list(self._active_servers)
        self.failed_servers = []
        self._failed_server_set = set()
        self.errors = {}

        servers_to_connect = self._servers_to_connect(self._all_servers)
        connected_servers: list[MCPServer] = []
        try:
            if self.connect_in_parallel:
                await self._connect_all_parallel(servers_to_connect)
            else:
                for server in servers_to_connect:
                    await self._attempt_connect(server)
                    if server not in self._failed_server_set:
                        connected_servers.append(server)
        except BaseException:
            if self.connect_in_parallel:
                await self._cleanup_servers(servers_to_connect)
            else:
                servers_to_cleanup = self._unique_servers(
                    [*connected_servers, *self.failed_servers]
                )
                await self._cleanup_servers(servers_to_cleanup)
            if self.drop_failed_servers:
                self._active_servers = [
                    server for server in self._all_servers if server in previous_connected_servers
                ]
            else:
                self._active_servers = previous_active_servers
            raise

        self._refresh_active_servers()

        return self._active_servers

    async def reconnect(self, *, failed_only: bool = True) -> list[MCPServer]:
        """Reconnect servers and return the active list.

        Args:
            failed_only: If True, only retry servers that previously failed.
                If False, cleanup and retry all servers.
        """
        if failed_only:
            servers_to_retry = self._unique_servers(self.failed_servers)
        else:
            await self.cleanup_all()
            servers_to_retry = list(self._all_servers)
            self.failed_servers = []
            self._failed_server_set = set()
            self.errors = {}

        servers_to_retry = self._servers_to_connect(servers_to_retry)
        try:
            if self.connect_in_parallel:
                await self._connect_all_parallel(servers_to_retry)
            else:
                for server in servers_to_retry:
                    await self._attempt_connect(server)
        finally:
            self._refresh_active_servers()
        return self._active_servers

    async def cleanup_all(self) -> None:
        """Cleanup all servers in reverse order."""
        for server in reversed(self._all_servers):
            try:
                await self._cleanup_server(server)
            except asyncio.CancelledError as exc:
                if not self.suppress_cancelled_error:
                    raise
                logger.debug(f"Cleanup cancelled for MCP server '{server.name}': {exc}")
                self.errors[server] = exc
            except Exception as exc:
                logger.exception(f"Failed to cleanup MCP server '{server.name}': {exc}")
                self.errors[server] = exc

    async def _run_with_timeout(
        self, func: Callable[[], Awaitable[Any]], timeout_seconds: float | None
    ) -> None:
        await _run_with_timeout_in_task(func, timeout_seconds)

    async def _attempt_connect(
        self, server: MCPServer, *, raise_on_error: bool | None = None
    ) -> None:
        if raise_on_error is None:
            raise_on_error = self.strict
        try:
            await self._run_connect(server)
            self._connected_servers.add(server)
            if server in self.failed_servers:
                self._remove_failed_server(server)
                self.errors.pop(server, None)
        except asyncio.CancelledError as exc:
            if not self.suppress_cancelled_error:
                raise
            self._record_failure(server, exc, phase="connect")
        except Exception as exc:
            self._record_failure(server, exc, phase="connect")
            if raise_on_error:
                raise
        except BaseException as exc:
            self._record_failure(server, exc, phase="connect")
            raise

    def _refresh_active_servers(self) -> None:
        if self.drop_failed_servers:
            failed = set(self._failed_server_set)
            self._active_servers = [server for server in self._all_servers if server not in failed]
        else:
            self._active_servers = list(self._all_servers)

    def _record_failure(self, server: MCPServer, exc: BaseException, phase: str) -> None:
        logger.exception(f"Failed to {phase} MCP server '{server.name}': {exc}")
        if server not in self._failed_server_set:
            self.failed_servers.append(server)
            self._failed_server_set.add(server)
        self.errors[server] = exc

    async def _run_connect(self, server: MCPServer) -> None:
        if self.connect_in_parallel:
            worker = self._get_worker(server)
            await worker.connect()
        else:
            await self._run_with_timeout(server.connect, self.connect_timeout_seconds)

    async def _cleanup_server(self, server: MCPServer) -> None:
        if self.connect_in_parallel and server in self._workers:
            worker = self._workers[server]
            if worker.is_done:
                self._workers.pop(server, None)
                self._connected_servers.discard(server)
                return
            try:
                await worker.cleanup()
            finally:
                self._workers.pop(server, None)
                self._connected_servers.discard(server)
            return
        try:
            await self._run_with_timeout(server.cleanup, self.cleanup_timeout_seconds)
        finally:
            self._connected_servers.discard(server)

    async def _cleanup_servers(self, servers: Iterable[MCPServer]) -> None:
        for server in reversed(list(servers)):
            try:
                await self._cleanup_server(server)
            except asyncio.CancelledError as exc:
                if not self.suppress_cancelled_error:
                    raise
                logger.debug(f"Cleanup cancelled for MCP server '{server.name}': {exc}")
                self.errors[server] = exc
            except Exception as exc:
                logger.exception(f"Failed to cleanup MCP server '{server.name}': {exc}")
                self.errors[server] = exc

    async def _connect_all_parallel(self, servers: list[MCPServer]) -> None:
        tasks = [
            asyncio.create_task(self._attempt_connect(server, raise_on_error=False))
            for server in servers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if not self.suppress_cancelled_error:
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    raise result
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result
        if self.strict and self.failed_servers:
            first_failure = None
            if self.suppress_cancelled_error:
                for server in self.failed_servers:
                    error = self.errors.get(server)
                    if error is None or isinstance(error, asyncio.CancelledError):
                        continue
                    first_failure = server
                    break
            else:
                first_failure = self.failed_servers[0]
            if first_failure is not None:
                error = self.errors.get(first_failure)
                if error is not None:
                    raise error
                raise RuntimeError(f"Failed to connect MCP server '{first_failure.name}'")

    def _get_worker(self, server: MCPServer) -> _ServerWorker:
        worker = self._workers.get(server)
        if worker is None or worker.is_done:
            worker = _ServerWorker(
                server=server,
                connect_timeout_seconds=self.connect_timeout_seconds,
                cleanup_timeout_seconds=self.cleanup_timeout_seconds,
            )
            self._workers[server] = worker
        return worker

    def _remove_failed_server(self, server: MCPServer) -> None:
        if server in self._failed_server_set:
            self._failed_server_set.remove(server)
        self.failed_servers = [
            failed_server for failed_server in self.failed_servers if failed_server != server
        ]

    def _servers_to_connect(self, servers: Iterable[MCPServer]) -> list[MCPServer]:
        unique = self._unique_servers(servers)
        if not self._connected_servers:
            return unique
        return [server for server in unique if server not in self._connected_servers]

    @staticmethod
    def _unique_servers(servers: Iterable[MCPServer]) -> list[MCPServer]:
        seen: set[MCPServer] = set()
        unique: list[MCPServer] = []
        for server in servers:
            if server not in seen:
                seen.add(server)
                unique.append(server)
        return unique
