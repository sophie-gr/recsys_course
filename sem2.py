"""
Семинар 2. Коллаборативная фильтрация
Цель: изучить user-based коллаборативную фильтрацию и построить
простую рекомендательную систему, которая предсказывает рейтинг и
рекомендует фильмы на основе похожих пользователей.

Задачи:
1. Реализовать вычисление сходства пользователей (Жаккар) по тем фильмам,
   которые они оба оценили.
2. Построить матрицу сходства пользователей с использованием матричных операций.
3. Предсказывать рейтинг пользователя для фильма с помощью top-k соседей.
4. Рекомендовать фильмы по оценкам ближайших похожих пользователей.

Алгоритмы (общее понимание):
- Жаккар считает схожесть как отношение размера пересечения к размеру объединения
  множеств просмотренных фильмов.
- User-based CF делает предсказание по взвешенному среднему рейтингам
  соседей, где веса — сходства пользователей.
- Для рекомендаций выбираем топ-R соседей, смотрим их высокие рейтинги
  (>=4.0) и рекомендуем топ-K фильмов, которые пользователь ещё не видел.
"""

from time import time

import numpy as np

from utils import build_user_item_matrix, id_to_movie

np.random.seed(42)


def jaccard_similarity(a: np.array, b: np.array) -> float:
    """
    Вычисление схожести пользователей по коэффициенту Жаккара.
    """
    mask_a = (a > 0).astype(int)
    mask_b = (b > 0).astype(int)
    
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    
    if union == 0:
        return 0.0
    return intersection / union

def build_user_user_matrix(user_item_matrix: np.ndarray) -> np.ndarray:
    """
    Вычисление матрицы сходств между пользователями
    по коэффициенту Жаккара.
    """
    X = (user_item_matrix > 0).astype(int)
    
    # Пересечение
    intersection = X @ X.T
    
    # Количество оцененных фильмов
    user_counts = X.sum(axis=1)
    
    # Объединение: |A| + |B| - |A ∩ B|
    sum_counts = user_counts[:, None] + user_counts[None, :]
    union = sum_counts - intersection
    
    # Деление с защитой от нуля
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(
            intersection, union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0
        )
    
    np.fill_diagonal(result, 1.0)
    return result

def predict_rating(
    user_id: int,
    item_id: int,
    user_user_matrix: np.ndarray,
    user_item_matrix: np.ndarray,
    topk: int = 10,
) -> float:
    """
    Предсказывает рейтинг для фильма item_id
    от пользователя user_id.
    """
    # Пользователи, оценившие фильм
    users_who_rated = user_item_matrix[:, item_id] > 0
    
    if not np.any(users_who_rated):
        return 0.0
    
    # Схожесть с активным пользователем
    similarities = user_user_matrix[user_id]
    sim_filtered = similarities[users_who_rated]
    ratings = user_item_matrix[users_who_rated, item_id]
    
    # Топ-k похожих
    if len(sim_filtered) > topk:
        top_indices = np.argsort(sim_filtered)[-topk:]
        sim_filtered = sim_filtered[top_indices]
        ratings = ratings[top_indices]
    
    sum_sim = np.sum(sim_filtered)
    if sum_sim == 0:
        return 0.0
    
    return np.sum(sim_filtered * ratings) / sum_sim

def predict_items_for_user(
    user_id: int,
    user_user_matrix: np.ndarray,
    user_item_matrix: np.ndarray,
    k: int = 5,
    r: int = 10,
) -> list:
    """
    Рекомендует фильмы пользователю на основе
    top-r похожих пользователей.
    """
    # Схожесть с другими пользователями
    similarities = user_user_matrix[user_id].copy()
    similarities[user_id] = -1
    
    # Топ-r похожих пользователей
    top_r_indices = np.argsort(similarities)[-r:]
    
    if len(top_r_indices) == 0:
        return []
    
    # Собираем все фильмы, которые соседи оценили >= 4.0
    candidate_movies = {}
    
    for neighbor_id in top_r_indices:
        neighbor_ratings = user_item_matrix[neighbor_id]
        # Находим фильмы с оценкой >= 4.0
        high_rated = np.where(neighbor_ratings >= 4.0)[0]
        
        for movie_id in high_rated:
            # Пропускаем фильмы, которые пользователь уже оценил
            if user_item_matrix[user_id, movie_id] > 0:
                continue
            # Добавляем в кандидаты
            if movie_id not in candidate_movies:
                candidate_movies[movie_id] = []
            candidate_movies[movie_id].append(neighbor_ratings[movie_id])
    
    if len(candidate_movies) == 0:
        return []
    
    # Считаем средний рейтинг для каждого кандидата
    movie_scores = []
    for movie_id, ratings in candidate_movies.items():
        avg_rating = np.mean(ratings)
        # Убеждаемся, что movie_id - это int
        movie_scores.append((int(movie_id), float(avg_rating)))
    
    # Сортируем по убыванию среднего рейтинга
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-k, убеждаясь что все элементы - int
    return [int(movie_id) for movie_id, _ in movie_scores[:k]]

if __name__ == "__main__":
    # Загрузка данных
    user_item_matrix = build_user_item_matrix()

    # Вычисление схожести между пользователями
    a, b = user_item_matrix[1], user_item_matrix[22]
    ab_sim = jaccard_similarity(a, b)
    print(f"Схожесть вкусов пользователей 1 и 2: {ab_sim:.2f}")

    tic = time()
    user_similarity_matrix = build_user_user_matrix(user_item_matrix)
    toc = time()
    print(f"Время вычисления матрицы сходства: {toc - tic:.2f} секунд")
    print(f"Размер матрицы сходства: {user_similarity_matrix.shape}")

    # Предсказание рейтинга фильма для пользователя
    user_id, item_id = 1, 47
    movie_name = id_to_movie(item_id)
    print(
        f"Предсказываем рейтинг фильма {item_id} - {movie_name} для пользователя {user_id}"
    )

    tic = time()
    item_rating = predict_rating(
        user_id, item_id, user_similarity_matrix, user_item_matrix
    )
    print(f"Предсказанный рейтинг фильма: {item_rating:.2f}")
    toc = time()
    print(f"Время предсказания рейтинга: {toc - tic:.2f} секунд")

    # Предсказание списка 5 фильмов с помощью коллаборативной фильтрации
    print("Предсказываем список из 5 фильмов для пользователя")
    tic = time()
    recomendations = predict_items_for_user(
        user_id, user_similarity_matrix, user_item_matrix
    )
    toc = time()
    print(f"Время предсказания рекомендаций: {toc - tic:.2f} секунд")
    print(f"Рекомендации для пользователя {user_id}: ")
    for movie_id in recomendations:
        score = predict_rating(
            user_id, movie_id, user_similarity_matrix, user_item_matrix
        )
        print(f"{id_to_movie(movie_id)} - {score:.2f}")

    # Предсказание списка 10 фильмов с помощью коллаборативной фильтрации
    print("Предсказываем список из 10 фильмов для пользователя")
    recomendations = predict_items_for_user(
        user_id, user_similarity_matrix, user_item_matrix, k=10
    )
    print(f"Рекомендации для пользователя {user_id}: ")
    for movie_id in recomendations:
        score = predict_rating(
            user_id, movie_id, user_similarity_matrix, user_item_matrix
        )
        print(f"{id_to_movie(movie_id)} - {score:.2f}")
