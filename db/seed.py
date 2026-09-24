"""Seed the database with a demo session so the UI has something to show on first run."""

from __future__ import annotations

import asyncio
import uuid

from db.database import AsyncSessionLocal
from db.models import Message, PendingAction, Session

DEMO_QUESTION = (
    "Research recent case law on force majeure clauses in commercial leases "
    "and draft a short SQL query to pull matching contracts from our database."
)


async def seed() -> None:
    async with AsyncSessionLocal() as db:
        session = Session(id=uuid.uuid4(), title="Force majeure research (demo)")
        db.add(session)
        await db.flush()

        db.add(
            Message(
                session_id=session.id,
                role="user",
                node=None,
                content=DEMO_QUESTION,
            )
        )
        db.add(
            Message(
                session_id=session.id,
                role="assistant",
                node="researcher",
                content=(
                    "Found 3 relevant cases discussing force majeure in commercial "
                    "leases [1][2][3]. Handing off to the coder agent to draft the SQL query."
                ),
            )
        )
        db.add(
            PendingAction(
                session_id=session.id,
                node_name="reviewer",
                payload={
                    "summary": "Draft SQL query ready for review before execution.",
                    "sql": "SELECT * FROM contracts WHERE clause_type = 'force_majeure';",
                },
                status="pending",
            )
        )

        await db.commit()
        print(f"Seeded demo session: {session.id}")


if __name__ == "__main__":
    asyncio.run(seed())
