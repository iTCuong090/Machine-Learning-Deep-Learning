Overview
Welcome to the ISE Training Challenge! In this competition, your goal is to build a machine learning model that predicts the music genre of a Spotify track using a set of numerical and categorical audio-related features. The competition dataset contains more than 73,000 unique tracks distributed across 112 genre classes. Each track belongs to exactly one genre. To make the task focus on learning from the musical characteristics of each track, identity-related metadata such as artist name, album name, and track name has been removed from the competition data.

Goal
Submissions will be evaluated using Macro F1 Score, meaning that each genre contributes equally to the final score regardless of how many samples it contains. The final ranking will be determined using the Private Leaderboard. Good luck, and have fun experimenting with different machine learning approaches!


Description
Music genre classification is a challenging machine learning problem because musical genres often share similar acoustic characteristics. Tracks from different genres may have comparable levels of energy, tempo, loudness, or danceability, while tracks within the same genre can still vary considerably.

In this competition, participants are asked to build a machine learning model that predicts the genre of a Spotify track from a set of audio and track-level characteristics.

The Challenge
Each track is represented by features describing different aspects of its musical characteristics, including duration, popularity, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, and time signature.

The target variable is track_genre.

Each track belongs to exactly one genre, making this a multi-class classification problem with 112 possible genre classes.

For every track in the test set, your model must predict one of the genre labels observed in the training data.

What Makes This Problem Interesting?
Classifying music genre from structured audio features is not straightforward.

Some genres may have distinctive acoustic patterns, while others can occupy very similar regions of the feature space. For example, multiple genres may contain tracks with similar energy, tempo, or danceability despite representing very different musical styles.

With 112 possible classes, a successful model must learn complex relationships between multiple features rather than relying on a single characteristic.

Participants are encouraged to explore different modeling strategies, feature engineering techniques, validation approaches, hyperparameter optimization, and ensemble methods to improve their predictions.

The challenge is not only to perform well on common genres, but also to build a model that generalizes consistently across the full range of genre classes.

Evaluation
Submissions are evaluated using the Macro F1 Score. The F1 score combines precision and recall into a single measure:
For this competition, an F1 score is calculated independently for each genre. The final score is the unweighted average of the F1 scores across all 112 genre classes:

F1 = 2 * Precision * Recall / (Precision + Recall)

Using Macro F1 means that every genre contributes equally to the final score, regardless of the number of samples belonging to that genre. A higher Macro F1 score is better. The final competition ranking is determined using the score on the Private Leaderboard.

Macro F1 = 1/112 * sum F1_i (i=1..112)

Submission File
For each track_id in the test set, you must predict exactly one value for track_genre. The submission file must contain exactly two columns:

track_id: the identifier of the track.
track_genre: the predicted genre. The file must contain a header and follow this format:
track_id,track_genre
0a1b2c3d4e,rock
1f2g3h4i5j,jazz
6k7l8m9n0o,hip-hop
Each track_id must appear exactly once, and every predicted track_genre must be one of the valid genre labels present in the training data. Refer to sample_submission.csv for the required submission structure.

Dataset Description
The competition data is provided in CSV format. Participants will use the training data to build their models and generate predictions for the unseen tracks in the test set.

Files
train.csv - the training dataset. Contains the input features and the target column track_genre.
test.csv - the test dataset. Contains the same input features as the training set, but does not include track_genre.
sample_submission.csv - an example submission file showing the required submission format.
genre_mapping.csv - mapping between the genre names used in train.csv and the integer genre_id values required for submission.
Columns
Identifier
track_id - unique identifier of the Spotify track. This column is used to match predictions with the corresponding rows in the test set.
popularity - popularity score of the track, ranging approximately from 0 to 100.
duration_ms - duration of the track in milliseconds.
explicit - indicates whether the track contains explicit content.
danceability - describes how suitable a track is for dancing based on musical elements such as tempo
energy - represents the perceived intensity and activity of a track.
key - estimated musical key of the track, represented using pitch class notation.
loudness - overall loudness of the track measured in decibels (dB).
mode - indicates whether the track is primarily in a major or minor mode.
speechiness - measures the presence of spoken words in the track.
acousticness - confidence measure indicating whether the track is acoustic. Values range from 0 to 1.
instrumentalness - estimates whether a track contains no vocals.
liveness - estimates the presence of an audience or live performance characteristics in the recording.
valence - describes the musical positiveness conveyed by a track.
tempo - estimated tempo of the track measured in beats per minute (BPM).
time_signature - estimated time signature of the track.
Target
track_genre - the genre label to be predicted. This column appears only in train.csv. The competition contains 112 possible genre classes, and each track in the competition belongs to exactly one class. For each row in the test set, participants must predict one genre.
Sample Submission
sample_submission.csv shows the required submission format:

track_id,track_genre
abc123,78
def456,45
ghi789,23
The order and values of track_id should match those provided in test.csv. Each prediction in track_genre must be one of the valid genre labels found in the training dataset.