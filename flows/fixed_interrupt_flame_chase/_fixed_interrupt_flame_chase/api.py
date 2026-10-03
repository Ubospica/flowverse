from __future__ import annotations

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowParams, LocalEnv
from pydantic import ConfigDict, Field, model_validator


class Agents(AgentCollection):
    first_chaser: Agent
    second_chaser: Agent


class Envs(EnvCollection):
    workspace: LocalEnv


class TurnAgents(AgentCollection):
    actor: Agent


class Params(FlowParams):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gate_config: str = Field(
        description="Absolute path to the trusted evaluator connection JSON."
    )
    run_dir: str = Field(
        description="New absolute controller output directory, outside the workspace."
    )
    max_valid_submissions_per_session: int = Field(default=5, ge=1, strict=True)
    active_time_limit_seconds: float = Field(default=21600, gt=0)
    review_reserve_seconds: float = Field(default=900, gt=0)
    review_turn_seconds: float = Field(default=600, gt=0)
    finalize_reserve_seconds: float = Field(default=60, gt=0)
    cleanup_reserve_seconds: float = Field(default=35, gt=0)
    poll_seconds: float = Field(default=0.25, gt=0, le=5)
    rest_seconds: float = Field(default=1, ge=0, le=60)

    @model_validator(mode="after")
    def review_fits(self) -> Params:
        owed = (
            self.review_turn_seconds
            + self.finalize_reserve_seconds
            + self.cleanup_reserve_seconds
        )
        if owed >= self.review_reserve_seconds:
            raise ValueError(
                "review reserve must exceed reviewer, cleanup and finalization time"
            )
        if self.review_reserve_seconds >= self.active_time_limit_seconds:
            raise ValueError("review reserve leaves no exploration time")
        return self
