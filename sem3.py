"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    """

    def __init__(self):
        self.embeddings = None
        self.movie_to_idx = {}  # movieId -> индекс в матрице
        self.idx_to_movie = {}  # индекс -> movieId
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        
        # Создаем маппинг movieId -> индекс
        unique_movies = self.movies_df["movieId"].unique()
        for idx, movie_id in enumerate(unique_movies):
            self.movie_to_idx[movie_id] = idx
            self.idx_to_movie[idx] = movie_id
        
        # Получаем жанры в правильном порядке
        genres_list = []
        for movie_id in unique_movies:
            genres = self.movies_df[self.movies_df["movieId"] == movie_id]["genres"].values[0]
            genres_list.append(genres)
        
        vectorizer = CountVectorizer(
            tokenizer=lambda s: s.split("|"),
            lowercase=False
        )
        # Строим матрицу эмбеддингов для фильмов
        embeddings_matrix = vectorizer.fit_transform(genres_list)
        self.embeddings = embeddings_matrix.toarray()

    def _get_embedding(self, item_id):
        """Получает эмбеддинг для movieId"""
        if item_id in self.movie_to_idx:
            idx = self.movie_to_idx[item_id]
            return self.embeddings[idx]
        return None

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.
        """
        # Вектор целевого фильма
        target_vec = self._get_embedding(item_id)
        if target_vec is None:
            return 0.0
        
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return 0.0
        
        # Находим фильмы, оцененные пользователем
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Считаем косинусное сходство с каждым оцененным фильмом
        similarities = []
        ratings_list = []
        
        for rated_movie_id in rated_items:
            rated_vec = self._get_embedding(rated_movie_id)
            if rated_vec is not None:
                rated_norm = np.linalg.norm(rated_vec)
                if rated_norm > 0:
                    sim = np.dot(target_vec, rated_vec) / (target_norm * rated_norm)
                    similarities.append(sim)
                    ratings_list.append(user_ratings[rated_movie_id])
        
        if len(similarities) == 0:
            return 0.0
        
        # Преобразуем в массивы numpy
        similarities = np.array(similarities)
        ratings_list = np.array(ratings_list)
        
        # Берем топ-k похожих
        if len(similarities) > k:
            top_indices = np.argsort(similarities)[-k:]
            similarities = similarities[top_indices]
            ratings_list = ratings_list[top_indices]
        
        # Взвешенное среднее
        sum_sim = np.sum(similarities)
        if sum_sim == 0:
            return 0.0
        
        predicted = np.sum(similarities * ratings_list) / sum_sim
        
        return np.clip(predicted, 0.0, 5.0)

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.
        """
        # Находим фильмы, оцененные пользователем
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []
        
        # Строим профиль пользователя (взвешенное среднее жанров)
        user_profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for movie_id in rated_items:
            rating = user_ratings[movie_id]
            item_vec = self._get_embedding(movie_id)
            if item_vec is not None:
                user_profile += rating * item_vec
                total_weight += rating
        
        if total_weight > 0:
            user_profile = user_profile / total_weight
        
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return []
        
        # Находим фильмы, которые пользователь НЕ оценил
        all_movie_ids = list(self.movie_to_idx.keys())
        rated_set = set(rated_items)
        unrated_items = [mid for mid in all_movie_ids if mid not in rated_set]
        
        if len(unrated_items) == 0:
            return []
        
        # Считаем сходство с профилем для всех неоцененных фильмов
        similarities = []
        for movie_id in unrated_items:
            item_vec = self._get_embedding(movie_id)
            if item_vec is not None:
                item_norm = np.linalg.norm(item_vec)
                if item_norm > 0:
                    sim = np.dot(user_profile, item_vec) / (profile_norm * item_norm)
                    similarities.append((int(movie_id), float(sim)))
                else:
                    similarities.append((int(movie_id), 0.0))
            else:
                similarities.append((int(movie_id), 0.0))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-n_recommendations (только int)
        return [movie_id for movie_id, _ in similarities[:n_recommendations]]


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")