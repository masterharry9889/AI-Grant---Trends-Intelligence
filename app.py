"""Streamlit UI for GrantBot."""
import streamlit as st

from Grant_agent import (create_agent, format_chat_history, load_chat_history,
                         load_vectorstore, generate_proposal, ClientProfile,
                         match_profile_to_categories)


def main() -> None:
    st.set_page_config(page_title="GrantBot", layout="wide")
    st.title("GrantBot — Grant Discovery & Proposal Assistant")

    with st.sidebar:
        st.header("Client Profile")
        profile = ClientProfile(
            organization=st.text_input("Organization"),
            contact_name=st.text_input("Contact name"),
            stage=st.selectbox("Stage", ["early-stage", "growth", "research", "nonprofit"]),
            industry=st.text_input("Industry"),
            location=st.text_input("Location", "India"),
            team_size=st.text_input("Team size"),
            funding_need=st.text_input("Funding need"),
            focus_areas=st.text_input("Focus areas"),
            beneficiaries=st.text_input("Beneficiaries"),
            impact_goals=st.text_input("Impact goals"),
            experience=st.text_input("Experience"),
        )

    st.header("Grant Query")
    user_query = st.text_area("Ask GrantBot", height=120)
    if st.button("Search Grants"):
        vectorstore = load_vectorstore()
        agent = create_agent(vectorstore)
        chat_history = load_chat_history(session_id="streamlit_session")
        formatted_history = format_chat_history(chat_history)

        try:
            response = agent.invoke({"input": user_query, "chat_history": formatted_history})
            answer = response["output"]
        except Exception as exc:
            st.error(f"Agent execution failed: {exc}")
            return

        st.subheader("GrantBot Results")
        st.write(answer)
        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(answer)

        st.subheader("Proposal Generator")
        grant_summary = st.text_area("Grant details summary", value=answer, height=220)
        if st.button("Generate Proposal Outline"):
            proposal = generate_proposal(grant_summary, profile)
            st.write(proposal)

        with st.expander("Suggested target categories"):
            categories = match_profile_to_categories(profile)
            st.write(categories)


if __name__ == "__main__":
    main()
