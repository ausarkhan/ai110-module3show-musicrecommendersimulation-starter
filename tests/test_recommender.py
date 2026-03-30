import os

from src.recommender import Song, UserProfile, Recommender
from src.recommender import load_songs, recommend_songs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_load_songs_converts_numeric_fields():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    assert len(songs) > 0
    first = songs[0]
    assert isinstance(first["id"], int)
    assert isinstance(first["energy"], float)
    assert isinstance(first["tempo_bpm"], float)


def test_profile_high_energy_pop():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.9}

    results = recommend_songs(user_prefs, songs, k=3)
    top_song, score, reasons = results[0]

    assert top_song["title"] == "Sunrise City"
    assert score > 0
    assert any("genre match" in reason for reason in reasons)


def test_profile_chill_lofi():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)
    user_prefs = {"genre": "lofi", "mood": "chill", "energy": 0.35}

    results = recommend_songs(user_prefs, songs, k=3)
    top_song, score, reasons = results[0]

    assert top_song["title"] == "Library Rain"
    assert score > 0
    assert any("mood match" in reason for reason in reasons)


def test_profile_intense_rock():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)
    user_prefs = {"genre": "rock", "mood": "intense", "energy": 0.92}

    results = recommend_songs(user_prefs, songs, k=3)
    top_song, score, reasons = results[0]

    assert top_song["title"] == "Storm Runner"
    assert score > 0
    assert any("energy similarity" in reason for reason in reasons)
