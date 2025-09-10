from pydantic import BaseModel as PydanticBaseModel, Field

class ApproachRecall(PydanticBaseModel):
    approach_name: str
    tutorial: str
    problem_description: str
    code: str
    planning: str

class Planning(PydanticBaseModel):
    planning: str

class CodeOutput(PydanticBaseModel):
    code: str = Field(..., description="Generated code to solve the problem")

class ProblemUnderstanding(PydanticBaseModel):
    understanding: str = Field(..., description="Understanding of the problem statement")

class Alignment(PydanticBaseModel):
    alignment_score: float = Field(..., description="Alignment score between plan and code")
    explanation: str = Field(..., description="Explanation of the alignment score")

class VerificationOutput(PydanticBaseModel):
    alignment_explanation: str = Field(default="No explanation provided", description="Alignment explanation for verification")
    alignment_score: float = Field(default=0.0, description="Alignment score for verification")
    coherence_explanation: str = Field(default="No explanation provided", description="Coherence explanation for verification")
    coherence_score: float = Field(default=0.0, description="Coherence score for verification")
    overall_solvability: int = Field(default=0, description="Overall solvability score")

class PlanAnalysisOutput(PydanticBaseModel):
    simulation: str = Field(default="No simulation provided", description="Step-by-step simulation of the plan")
    insights: str = Field(default="No insights provided", description="Error localization or problems in the plan")
    pr_tok: int = Field(default=0, description="Prompt tokens used")
    com_tok: int = Field(default=0, description="Completion tokens used")

class CodeAnalysisOutput(PydanticBaseModel):
    simulation: str = Field(default="No simulation provided", description="Line-by-line simulation of the code execution to identify errors")
    insights: str = Field(default="No insights provided", description="Error localization or problems in the code")
    pr_tok: int = Field(default=0, description="Prompt tokens used")
    com_tok: int = Field(default=0, description="Completion tokens used")

class ContentAnalysisOutput(PydanticBaseModel):
    plan_code_insights: str = Field(default="No insights provided", description="Plan-code alignment insights")
    pr_tok: int = Field(default=0, description="Prompt tokens used")
    com_tok: int = Field(default=0, description="Completion tokens used")

class ConfidenceOutput(PydanticBaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="No reasoning provided", description="Reasoning for the confidence score")

class ConsistencyOutput(PydanticBaseModel):
    consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="Consistency score")
    reasoning: str = Field(default="No reasoning provided", description="Reasoning for the consistency score")

class CodeGenerationOutput(PydanticBaseModel):
    best_code: str = Field(default="", description="Generated code")
    flag: bool = Field(default=False, description="Whether the code passed sample tests")
    test_log: str = Field(default="", description="Test log from evaluation")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score based on test cases")
    pr_tok: int = Field(default=0, description="Prompt tokens used")
    com_tok: int = Field(default=0, description="Completion tokens used")

class PlanOutput(PydanticBaseModel):
    planning: str = Field(..., description="Planning steps")
    code: str = Field(..., description="Best code for the plan")
    llm_score: int = Field(..., description="LLM-verified solvability score")
    code_score: float = Field(..., ge=0.0, le=1.0, description="Code evaluation score")
    test_log: str = Field(default="", description="Test log from evaluation")