# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for attaching A2A endpoints to an existing FastAPI application.

This module centralises the boilerplate needed to expose Agent2Agent (A2A)
endpoints from a scaffolded project. Calling :func:`attach_a2a_routes`
registers the dynamic agent-card endpoint and the JSON-RPC endpoint on a
caller-supplied FastAPI app, so the same app can serve both the standard ADK
web routes and the A2A routes without duplicating the wiring in every
generated project.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskStore
from a2a.types import AgentCapabilities, AgentExtension
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

if TYPE_CHECKING:
    from fastapi import FastAPI
    from google.adk.agents import BaseAgent
    from google.adk.runners import Runner

# URI advertised on the agent card describing the executor extension shipped
# by ADK. Kept as a module-level constant so callers can override or extend
# the capabilities list when needed.
_ADK_AGENT_EXECUTOR_EXTENSION_URI = "https://google.github.io/adk-docs/a2a/a2a-extension/"


def _default_capabilities() -> AgentCapabilities:
    """Returns the default A2A capabilities used by scaffolded projects."""
    return AgentCapabilities(
        streaming=True,
        extensions=[
            AgentExtension(
                uri=_ADK_AGENT_EXECUTOR_EXTENSION_URI,
                description=("Ability to use the new agent executor implementation"),
            ),
        ],
    )


async def attach_a2a_routes(
    app: FastAPI,
    *,
    agent: BaseAgent,
    runner: Runner,
    task_store: TaskStore,
    rpc_path: str,
    capabilities: AgentCapabilities | None = None,
    agent_version: str | None = None,
    app_url: str | None = None,
) -> None:
    """Attach A2A endpoints to an existing FastAPI app.

    Builds a dynamic agent card from ``agent`` and registers the A2A
    JSON-RPC endpoint, the well-known agent-card endpoint, and the extended
    agent-card endpoint under ``rpc_path`` on the caller-supplied ``app``.

    Args:
        app: The FastAPI application the A2A routes should be attached to.
        agent: The root ADK agent that the agent card is built from.
        runner: The ADK runner used by the A2A executor. The caller is
            responsible for constructing this with the desired session,
            artifact, and memory services so the A2A path observes the
            same backends as the standard ADK path.
        task_store: The :class:`a2a.server.tasks.TaskStore` implementation
            used by the A2A request handler. Pass an
            :class:`~a2a.server.tasks.InMemoryTaskStore` for local
            development and a managed/persistent store for production
            deployments.
        rpc_path: The path prefix at which the A2A routes are mounted, for
            example ``"/a2a/my_agent"``. Used for both the JSON-RPC URL
            and as the prefix for the agent-card URLs.
        capabilities: Optional override for the A2A capabilities advertised
            on the agent card. Defaults to streaming + the ADK executor
            extension.
        agent_version: Optional explicit version string published on the
            agent card. Defaults to the ``AGENT_VERSION`` environment
            variable or ``"0.1.0"``.
        app_url: Optional explicit public URL the agent is reachable at.
            Defaults to the ``APP_URL`` environment variable or
            ``http://0.0.0.0:8000``. The agent card's ``url`` field is set
            to ``{app_url}{rpc_path}`` per the A2A specification.

    This function is idempotent only with respect to the agent-card build;
    repeated calls would register duplicate routes on ``app``. Callers
    should invoke it once per app, typically inside a FastAPI ``lifespan``
    context manager so the agent card can be built asynchronously.
    """
    resolved_app_url = app_url or os.getenv("APP_URL", "http://0.0.0.0:8000")
    resolved_agent_version = agent_version or os.getenv("AGENT_VERSION", "0.1.0")
    resolved_capabilities = capabilities or _default_capabilities()

    agent_card = await AgentCardBuilder(
        agent=agent,
        capabilities=resolved_capabilities,
        rpc_url=f"{resolved_app_url}{rpc_path}",
        agent_version=resolved_agent_version,
    ).build()

    request_handler = DefaultRequestHandler(
        agent_executor=A2aAgentExecutor(runner=runner),
        task_store=task_store,
    )

    a2a_app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)
    a2a_app.add_routes_to_app(
        app,
        agent_card_url=f"{rpc_path}{AGENT_CARD_WELL_KNOWN_PATH}",
        rpc_url=rpc_path,
        extended_agent_card_url=f"{rpc_path}{EXTENDED_AGENT_CARD_PATH}",
    )
