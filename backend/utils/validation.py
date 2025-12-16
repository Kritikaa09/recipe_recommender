import re
from fastapi import HTTPException

def validate_query(q: str):
    q = q.strip().lower()

    # Too short → useless
    if len(q) < 3:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid food name or ingredient"
        )

    # No alphabets → garbage
    if not re.search(r"[a-z]", q):
        raise HTTPException(
            status_code=400,
            detail="Invalid input"
        )

    return q
