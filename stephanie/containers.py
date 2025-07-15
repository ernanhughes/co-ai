from dependency_injector import containers, providers
from stephanie.protocols.sample_protocols import DirectAnswerProtocol
from stephanie.protocols.sample_protocols import CodeExecutionProtocol
from stephanie.agents.mrq_strategy import MRQStrategyAgent
from stephanie.memory.pipeline_stage_store import PipelineStageStore
from sqlalchemy.orm import Session
from stephanie.agents.g3ps_solver import G3PSSolverAgent


class AppContainer(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()

    # Database session
    session_factory = providers.Factory(Session)

    # Services
    pipeline_stage_store = providers.Singleton(
        PipelineStageStore,
        session=session_factory
    )

    # Protocols
    direct_answer_protocol = providers.Singleton(DirectAnswerProtocol)
    code_execution_protocol = providers.Singleton(CodeExecutionProtocol)

    # Agent factories
    mrq_strategy_agent = providers.Factory(
        MRQStrategyAgent,
        cfg=config.agent.mrq,
        memory=None,  # You can wire this with MemoryTool if needed
        logger=None
    )

    g3ps_solver_agent = providers.Factory(
        G3PSSolverAgent,
        protocol=direct_answer_protocol,
        cfg=config.agent.g3ps,
        logger=None
    )