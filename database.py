import sqlite3
from datetime import datetime


DB_NAME = "youtube_notes.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Store information about each YouTube video
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS videos (
                                                         video_id TEXT PRIMARY KEY,
                                                         title TEXT NOT NULL,
                                                         url TEXT NOT NULL,
                                                         created_at TEXT NOT NULL,
                                                         updated_at TEXT NOT NULL
                   )
                   """)

    # Store every generated version of the notes
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS notes (
                                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                        video_id TEXT NOT NULL,
                                                        content TEXT NOT NULL,
                                                        version INTEGER NOT NULL,
                                                        created_at TEXT NOT NULL,
                                                        is_current INTEGER NOT NULL DEFAULT 1,

                                                        FOREIGN KEY (video_id)
                       REFERENCES videos(video_id)
                       )
                   """)

    connection.commit()
    connection.close()


def save_notes(video_id, title, url, content):
    connection = get_connection()
    cursor = connection.cursor()

    now = datetime.now().isoformat(timespec="seconds")

    # Check if this YouTube video already exists
    cursor.execute(
        """
        SELECT video_id
        FROM videos
        WHERE video_id = ?
        """,
        (video_id,)
    )

    video_exists = cursor.fetchone()

    if video_exists is None:

        # First time generating notes for this video
        cursor.execute(
            """
            INSERT INTO videos
                (video_id, title, url, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                video_id,
                title,
                url,
                now,
                now
            )
        )

        version = 1

    else:

        # Find the latest version number
        cursor.execute(
            """
            SELECT MAX(version)
            FROM notes
            WHERE video_id = ?
            """,
            (video_id,)
        )

        result = cursor.fetchone()

        version = (result[0] or 0) + 1

        # Mark the previous notes as no longer current
        cursor.execute(
            """
            UPDATE notes
            SET is_current = 0
            WHERE video_id = ?
              AND is_current = 1
            """,
            (video_id,)
        )

        # Update video metadata
        cursor.execute(
            """
            UPDATE videos
            SET title = ?,
                url = ?,
                updated_at = ?
            WHERE video_id = ?
            """,
            (
                title,
                url,
                now,
                video_id
            )
        )

    # Insert the new notes
    cursor.execute(
        """
        INSERT INTO notes
            (video_id, content, version, created_at, is_current)
        VALUES (?, ?, ?, ?, 1)
        """,
        (
            video_id,
            content,
            version,
            now
        )
    )

    connection.commit()
    connection.close()

    return version


def load_notes(video_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT content
        FROM notes
        WHERE video_id = ?
          AND is_current = 1
        """,
        (video_id,)
    )

    result = cursor.fetchone()

    connection.close()

    if result is None:
        return None

    return result[0]


def load_metadata(video_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT video_id, title, url, created_at, updated_at
        FROM videos
        WHERE video_id = ?
        """,
        (video_id,)
    )

    result = cursor.fetchone()

    connection.close()

    if result is None:
        return None

    return {
        "video_id": result[0],
        "title": result[1],
        "url": result[2],
        "created_at": result[3],
        "updated_at": result[4]
    }


def get_all_notes():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            video_id,
            title,
            url,
            created_at,
            updated_at
        FROM videos
        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    notes = []

    for row in rows:
        notes.append({
            "video_id": row[0],
            "title": row[1],
            "url": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        })

    return notes


def get_versions(video_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            content,
            version,
            created_at,
            is_current
        FROM notes
        WHERE video_id = ?
        ORDER BY version DESC
        """,
        (video_id,)
    )

    rows = cursor.fetchall()

    connection.close()

    versions = []

    for row in rows:
        versions.append({
            "id": row[0],
            "content": row[1],
            "version": row[2],
            "created_at": row[3],
            "is_current": bool(row[4])
        })

    return versions


def revert_notes(video_id, version):
    connection = get_connection()
    cursor = connection.cursor()

    # Check that the requested version exists
    cursor.execute(
        """
        SELECT id
        FROM notes
        WHERE video_id = ?
          AND version = ?
        """,
        (
            video_id,
            version
        )
    )

    requested_version = cursor.fetchone()

    if requested_version is None:
        connection.close()
        return False

    # Remove current status from all versions
    cursor.execute(
        """
        UPDATE notes
        SET is_current = 0
        WHERE video_id = ?
        """,
        (video_id,)
    )

    # Make the selected version current
    cursor.execute(
        """
        UPDATE notes
        SET is_current = 1
        WHERE video_id = ?
          AND version = ?
        """,
        (
            video_id,
            version
        )
    )

    # Update video's last-modified time
    now = datetime.now().isoformat(timespec="seconds")

    cursor.execute(
        """
        UPDATE videos
        SET updated_at = ?
        WHERE video_id = ?
        """,
        (
            now,
            video_id
        )
    )

    connection.commit()
    connection.close()

    return True


if __name__ == "__main__":

    initialize_database()

    print("Database initialized.")

    # Test saving notes
    version = save_notes(
        "ut-ZwrpPb0U",
        "ADLC and Harness Engineering",
        "https://www.youtube.com/watch?v=ut-ZwrpPb0U",
        """# ADLC and Harness Engineering

- ADLC represents the Agentic Development Life Cycle.
- The three Hs are Harness, Handoffs, and Humans.
"""
    )

    print("Saved version:", version)

    # Test current notes
    print("\nCurrent notes:")
    print(load_notes("ut-ZwrpPb0U"))

    # Test metadata
    print("\nMetadata:")
    print(load_metadata("ut-ZwrpPb0U"))

    # Test all videos
    print("\nAll saved videos:")
    print(get_all_notes())

    # Test versions
    print("\nVersions:")
    print(get_versions("ut-ZwrpPb0U"))