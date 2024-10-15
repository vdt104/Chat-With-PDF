from app.chat.redis import client



def score_conversation(
    conversation_id: str, score: float, llm: str, retriever: str, memory: str
) -> None:
    """
    This function interfaces with langfuse to assign a score to a conversation, specified by its ID.
    It creates a new langfuse score utilizing the provided llm, retriever, and memory components.
    The details are encapsulated in JSON format and submitted along with the conversation_id and the score.

    :param conversation_id: The unique identifier for the conversation to be scored.
    :param score: The score assigned to the conversation.
    :param llm: The Language Model component information.
    :param retriever: The Retriever component information.
    :param memory: The Memory component information.

    Example Usage:

    score_conversation('abc123', 0.75, 'llm_info', 'retriever_info', 'memory_info')
    """

    score = min(max(score, 0), 1)

    client.hincrby("llm_score_values", llm, score)
    client.hincrby("llm_score_counts", llm, 1)

    client.hincrby("retriever_score_values", retriever, score)
    client.hincrby("retriever_score_counts", retriever, 1)


def get_scores():
    """
    Retrieves and organizes scores from the langfuse client for different component types and names.
    The scores are categorized and aggregated in a nested dictionary format where the outer key represents
    the component type and the inner key represents the component name, with each score listed in an array.

    The function accesses the langfuse client's score endpoint to obtain scores.
    If the score name cannot be parsed into JSON, it is skipped.

    :return: A dictionary organized by component type and name, containing arrays of scores.

    Example:

        {
            'llm': {
                'chatopenai-3.5-turbo': [avg_score1],
                'chatopenai-4': [avg_score2]
            },
            'retriever': { 'pinecone_store': [avg_score3]},
        }
    """

    aggregate = {"llm": {}, "retriever": {}}

    for component_type in aggregate.keys():
        values = client.hgetall(f"{component_type}_score_values")
        counts = client.hgetall(f"{component_type}_score_counts")

        names = values.keys()

        for name in names:
            score = int(values.get(name, 1))
            count = int(counts.get(name, 1))
            avg = score / count
            aggregate[component_type][name] = [avg]

    return aggregate