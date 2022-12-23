from transformers import pipeline


def answer_question(question, context):
    """
    Question Answering
    :return:
    """
    # Initialise the QA pipeline
    qa = pipeline("question-answering")

    # Generating an answer to the question in context
    answer = qa(question=question, context=context)

    # Print the answer
    # print(f"Question: {question}")
    print(f"Answer: '{answer['answer']}'")

    return answer['answer']


if __name__ == "__main__":
    Q = "What is the object"
    ans = answer_question(Q, "this is dog")
    print(ans)

    # answer_question()
