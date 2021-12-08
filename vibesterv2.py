import os
from itertools import permutations

import numpy as np
import pandas as pd
import sklearn
import spotipy
from sklearn.cluster import KMeans as kp
from sklearn.decomposition import PCA
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from kneed import KneeLocator


def get_track_ids(playlist_items):
    track_ids = []
    for item in playlist_items['items']:
        track_ids.append(item['track']['id'])
    return track_ids


def get_track_audio_features(tracks, sp):
    audio_features = []
    # audio_features = pd.DataFrame()
    for track in tracks:
        features = sp.audio_features(track)
        danceability = features[0]['danceability']
        energy = features[0]['energy']
        ky = features[0]['key']
        loudness = features[0]['loudness']
        mde = features[0]['mode']
        speechiness = features[0]['speechiness']
        tempo = features[0]['tempo']
        track_id = features[0]['id']
        valence = features[0]['valence']
        instrumentalness = features[0]['instrumentalness']
        track_info = [track_id, danceability, energy, ky, loudness, mde, speechiness, tempo, valence, instrumentalness]
        audio_features.append(track_info)
    return audio_features


def get_playlist_ids(playlists):
    playlist_ids = []
    for playlist in playlists['items']:
        playlist_ids.append(playlist['id'])
    return playlist_ids


def get_track_features(track_id, sp):
    meta = sp.track(track_id)
    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    spotify_url = meta['external_urls']['spotify']
    album_cover = meta['album']['images'][0]['url']
    track_info = [name, album, artist, spotify_url, album_cover]
    return track_info


def get_current_user(sp):
    meta = sp.me()
    # meta
    name = meta['display_name']
    return name


def get_playlist_features(playlist_id, sp):
    display_name = get_current_user(sp)

    meta = sp.playlist(playlist_id)
    # meta
    id = meta['id']
    name = meta['name']
    owner = meta['owner']['display_name']
    public = meta['public']
    num_tracks = meta['tracks']['total']
    num_followers = meta['followers']['total']
    track_info = [id, name, owner, public, num_tracks, num_followers]

    if owner == display_name:
        return track_info


def unload_to_df(features):
    column_names = ["track_id", "danceability", "energy", "key", "loudness", "mode", "speechiness", "tempo", "valence",
                    "instrumentalness"]
    features_df = pd.DataFrame(data=features, columns=column_names)

    return features_df


def cluster_tracks(df):
    scaler = MinMaxScaler()
    df[["danceability", "energy", "key", "loudness", "mode", "speechiness", "tempo", "valence",
        "instrumentalness"]] = scaler.fit_transform(df[["danceability", "energy", "key", "loudness", "mode",
                                                        "speechiness", "tempo", "valence", "instrumentalness"]])
    columns = df.columns[np.r_[1:8]]
    normalized_data = df[columns]
    normalized_data = pd.DataFrame(normalized_data)

    KMeans = kp(n_clusters=4).fit(normalized_data)
    df['cluster'] = KMeans.fit_predict(normalized_data)
    tracks_clustered = df.sort_values('track_id').reset_index(drop=True)

    KMeans = kp(n_clusters=5).fit(normalized_data)
    df['cluster_detailed'] = KMeans.fit_predict(normalized_data)
    tracks_clustered_again = df.sort_values('track_id').reset_index(drop=True)

    KMeans = kp(n_clusters=8).fit(normalized_data)
    df['cluster_super_detailed'] = KMeans.fit_predict(normalized_data)
    tracks_clustered_again_again = df.sort_values('track_id').reset_index(drop=True)
    tracks_clustered_again_again = tracks_clustered_again_again.sort_values('cluster_super_detailed')
    tracks_clustered_again_again = tracks_clustered_again_again.sort_values('cluster')
    tracks_clustered_again_again = tracks_clustered_again_again.sort_values('cluster_detailed')

    return tracks_clustered_again_again


class FindPlaylists:
    def __init__(self):
        self.SPOTIPY_CLIENT_ID = os.environ["client_id"]
        self.SPOTIPY_CLIENT_SECRET = os.environ["secret"]
        self.SPOTIPY_REDIRECT_URI = os.environ["uri"]
        self.SCOPE = "playlist-modify-public playlist-modify-private user-library-read"

        pd.options.display.width = None
        pd.options.display.max_columns = None
        pd.set_option('display.max_rows', 3000)
        pd.set_option('display.max_columns', 3000)

    def connect_to_spotify(self):
        conn = spotipy.Spotify(
            auth_manager=SpotifyOAuth(client_id=self.SPOTIPY_CLIENT_ID, client_secret=self.SPOTIPY_CLIENT_SECRET,
                                      redirect_uri=self.SPOTIPY_REDIRECT_URI, scope=self.SCOPE))

        return conn

    def get_playlist(self):
        sp = self.connect_to_spotify()
        users_playlists = sp.current_user_playlists(limit=50, offset=0)
        playlist_ids = get_playlist_ids(users_playlists)
        my_playlists = []
        for i in range(len(playlist_ids)):
            # time.sleep(.5)
            playlist = get_playlist_features(playlist_ids[i], sp)
            if playlist is not None:
                my_playlists.append(playlist)
        print(my_playlists)
        # selected_playlist = input("Enter playlist ID:")
        selected_playlist = '5eU3gUPR7j72w3DzpjpxjR'
        current_user = get_current_user(sp)
        playlist_items = sp.user_playlist_tracks(current_user, selected_playlist, 'items(track(id))', limit=100)
        track_ids = get_track_ids(playlist_items)
        features = get_track_audio_features(track_ids, sp)
        df = unload_to_df(features)  # create df with audio features for the songs
        df['row_num'] = np.arange(len(df))
        print(df)
        clustered_df = cluster_tracks(df)
        clustered_df['new_row_num'] = np.arange(len(clustered_df))
        print(clustered_df)
        clustered_list = clustered_df.values.tolist()
        print(clustered_list)
        for i in range(len(clustered_list)):
            track_to_be_moved = clustered_list[i][10]
            sp.playlist_reorder_items(selected_playlist, track_to_be_moved, i)
            print("moving to next song")
        # reorder tracks

    def perform_operations(self):
        try:
            self.get_playlist()
        except Exception as e:
            f"Failed with the error {repr(e)}"


if __name__ == "__main__":
    SpotifyFind = FindPlaylists()
    SpotifyFind.perform_operations()
