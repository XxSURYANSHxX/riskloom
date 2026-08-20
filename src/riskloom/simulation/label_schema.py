from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from riskloom.simulation.event_schema import EventId

CampaignId = Annotated[str, StringConstraints(pattern=r"^cmp_[0-9a-f]{32}$")]
ScenarioInstanceId = Annotated[str, StringConstraints(pattern=r"^scn_[0-9a-f]{32}$")]


class SplitName(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class ScenarioType(StrEnum):
    NORMAL = "normal"
    LEGITIMATE_RETRY = "legitimate_retry"
    FLASH_SALE = "flash_sale"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"
    LEGITIMATE_FAILURE = "legitimate_failure"
    CARD_TESTING_CAMPAIGN = "card_testing_campaign"


class CampaignProfile(StrEnum):
    BASELINE_REUSE = "baseline_reuse"
    ENTITY_REUSE_SHIFT = "entity_reuse_shift"


class GeneratorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    component_version: Literal["1.0.0"] = "1.0.0"
    scenario_instance_id: ScenarioInstanceId | None
    campaign_profile: CampaignProfile | None


class GroundTruthLabel(BaseModel):
    """Evaluation-only truth that must never enter model-visible artifacts or replay."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    event_id: EventId
    split: SplitName
    is_attack: bool
    campaign_id: CampaignId | None
    scenario_type: ScenarioType
    generator_metadata: GeneratorMetadata

    @model_validator(mode="after")
    def validate_attack_fields(self) -> Self:
        if self.is_attack:
            if self.scenario_type is not ScenarioType.CARD_TESTING_CAMPAIGN:
                raise ValueError("attack labels must use the campaign scenario")
            if self.campaign_id is None or self.generator_metadata.campaign_profile is None:
                raise ValueError("attack labels require campaign metadata")
        else:
            if self.scenario_type is ScenarioType.CARD_TESTING_CAMPAIGN:
                raise ValueError("campaign scenarios must be attack labels")
            if self.campaign_id is not None or self.generator_metadata.campaign_profile is not None:
                raise ValueError("legitimate labels must not include campaign metadata")
        return self
