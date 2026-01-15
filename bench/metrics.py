from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

def mk_correctness_0_2():
    return GEval(
        name="Correctness (0-2)",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Compare the actual output against the expected output (rules/ground truth).",
            "Primary criterion: the bot must not invent or contradict key facts/rules/points.",
            "Minor omissions are acceptable only if they don’t change the correctness of the main answer.",
            "Assign an integer score using the rubric."
        ],
        rubric=[
            Rubric(score_range=(0, 0), expected_outcome="Completely incorrect; contradicts the expected output or invents key facts/rules/points."),
            Rubric(score_range=(1, 1), expected_outcome="Partially correct; some correct info but missing or wrong important parts; no major fabricated facts."),
            Rubric(score_range=(2, 2), expected_outcome="Fully correct and consistent with the expected output; no fabricated facts/rules/points.")
        ],
        threshold=0.5,
    )

def mk_faithfulness_0_2():
    return GEval(
        name="Faithfulness / Groundness (0-2)",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        evaluation_steps=[
            "Use ONLY the retrieval context as the source of truth.",
            "Check whether statements in the actual output are supported by the retrieval context.",
            "Penalize hallucinations: claims, rules, numbers, or facts not present in the retrieval context.",
            "Assign an integer score using the rubric.",
        ],
        rubric=[
            Rubric(
                score_range=(0, 0),
                expected_outcome="Completely ungrounded: the answer is mostly invented / not supported by the retrieval context (full hallucination).",
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="Partially grounded: there are hallucinated or unsupported claims, but some parts rely on the retrieval context.",
            ),
            Rubric(
                score_range=(2, 2),
                expected_outcome="Fully grounded: all factual claims are supported by the retrieval context; no hallucinated rules/facts/points.",
            ),
        ],
        threshold=0.5,
    )

def mk_completeness_0_2():
    return GEval(
        name="Completeness (0-2)",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=[
            "Identify the distinct questions/sub-questions and required aspects in the input.",
            "Check whether the actual output addresses each required aspect with a concrete answer (not just restating the question).",
            "Do not penalize style; evaluate only coverage of the necessary aspects.",
            "Assign an integer score using the rubric.",
        ],
        rubric=[
            Rubric(
                score_range=(0, 0),
                expected_outcome="Did not answer any of the questions/aspects from the input; mostly irrelevant or empty.",
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="Answered at least one part, but coverage is incomplete: missing important aspects/questions.",
            ),
            Rubric(
                score_range=(2, 2),
                expected_outcome="Fully answered all questions/aspects from the input with adequate coverage.",
            ),
        ],
        threshold=0.5,
    )

def mk_brevity_0_2():
    return GEval(
        name="Brevity / Non-verbosity (0-2)",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=[
            "Determine what information is necessary to answer the input well.",
            "Check whether the actual output contains unnecessary filler, repetition, or off-topic content.",
            "Do NOT penalize needed clarifications or safety caveats; only penalize excess verbosity beyond what is useful.",
            "Assign an integer score using the rubric.",
        ],
        rubric=[
            Rubric(
                score_range=(0, 0),
                expected_outcome="Too verbose: lots of filler/water, repetition, or off-topic details; the core answer is buried.",
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="Some verbosity: mostly on-topic but includes noticeable extra text that could be removed without losing key meaning.",
            ),
            Rubric(
                score_range=(2, 2),
                expected_outcome="Concise and to the point: minimal necessary text; no filler; directly answers the question.",
            ),
        ],
        threshold=0.5,
    )