import re
import pandas as pd
import time
import logging
from config import nlp, api

def is_valid_clue(clue):
    notes_to_remove = ["description acceptable", "two answers required"]
    return not any(note.lower() in clue.lower() for note in notes_to_remove)

def clean_answer(answer):
    return re.sub(r"\(.*?\)|\[.*?\]", "", answer, flags=re.DOTALL).strip()

def process_answerline_input(
    answer,
    searchType="answer",
    exactPhrase=False,
    ignoreWordOrder=False,
    removeParentheticals=True,
    setName="",
    difficulties=None,
    categories="Literature"
):
    """
    Query qbreader with given parameters, optionally strip parentheticals,
    extract tossup clues, compute placement, and return a DataFrame.
    """
    try:
        logging.info(f"Processing '{answer}' [{searchType}] removeParentheticals={removeParentheticals}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = api.query(
                    questionType="tossup",
                    searchType=searchType,
                    queryString=answer,
                    exactPhrase=exactPhrase,
                    ignoreWordOrder=ignoreWordOrder,
                    setName=setName,
                    difficulties=difficulties or [],
                    categories=categories,
                    maxReturnLength=500
                )
                break
            except Exception as e:
                logging.error(f"Query error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        else:
            logging.error(f"Failed to query after {max_retries} attempts.")
            return None

        clues = []
        for tossup in resp.tossups:
            raw_ans = tossup.answer_sanitized.strip()
            # apply or skip cleaning
            cleaned_ans = clean_answer(raw_ans) if removeParentheticals else raw_ans
            if searchType == "answer" and answer.lower() not in cleaned_ans.lower():
                continue

            doc = nlp(tossup.question_sanitized)
            sents = [s.text.strip() for s in doc.sents]
            if len(sents) > 1:
                sents.pop()  # drop giveaway
            total = len(sents)
            if total == 0:
                continue
            for i, clue in enumerate(sents):
                pct = round((i / total) * 100, 2)
                if is_valid_clue(clue):
                    clues.append([clue, pct, cleaned_ans])

        if not clues:
            logging.warning("No valid clues found.")
            return None

        return pd.DataFrame(clues, columns=["clue", "placement", "answerline"])

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None
