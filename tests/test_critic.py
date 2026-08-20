from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_critic_validates_final_citations() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Compare agent architectures"),
        sources=[
            SourceDocument(
                title="Source",
                snippet="Evidence",
                metadata={"citation_label": "source"},
            )
        ],
        final_answer="Grounded answer [source].",
    )

    CriticAgent().run(state)

    assert state.agent_results[-1].agent == AgentName.CRITIC
    assert state.agent_results[-1].metadata["passed"] is True
    assert state.trace[-1]["name"] == "critic"
