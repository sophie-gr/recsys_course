"""
Семинар 1. Простые рекомендации
Цель: построить базовые рекомендательные методы и оценить качество
на примере мобильной системы MovieLens.

Задачи:
1) Реализовать случайные рекомендации.
2) Реализовать рекомендации популярных фильмов на основе средних рейтингов.
3) Оценить системы по точности попадания в исторические оценки пользователя.

Для каждого метода требуется реализовать функцию, возвращающую
набор рекомендаций и метрику accuracy.
"""

import numpy as np

from utils import load_data


def random_recommend(n_recommendations: int = 10, seed: int = 42) -> list[int]:
    """
    Рекомендует случайные фильмы из всех доступных.

    Алгоритм:
    1) Загружаем рейтинговый DataFrame.
    2) Берём уникальные ID фильмов.
    3) Случайно выбираем n фильмов без повторов.

    Args:
        n_recommendations: Количество рекомендаций.
        seed: Seed для воспроизводимости.

    Returns:
        Список ID фильмов.
    """
    ratings_df, _ = load_data()
    np.random.seed(seed)
    all_movie_ids = ratings_df["movieId"].unique()
    recommendations = np.random.choice(
        all_movie_ids, size=n_recommendations, replace=False
    )
    return recommendations.tolist()


def top_n_recommend(
    n_recommendations: int = 10, min_ratings: int = 10
) -> list[tuple[int, float, int, str]]:
    """
    Рекомендует самые популярные фильмы по средней оценке и количеству оценок.
    """
    # Загружаем данные
    ratings_df, movies_df = load_data()
    
    # Группируем по movieId и считаем средний рейтинг и число оценок
    movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])
    movie_stats.columns = ['avg_rating', 'rating_count']
    
    # Фильтруем фильмы с rating_count >= min_ratings
    filtered = movie_stats[movie_stats['rating_count'] >= min_ratings]
    
    # Сортируем по avg_rating и rating_count по убыванию
    sorted_movies = filtered.sort_values(['avg_rating', 'rating_count'], ascending=False)
    
    # Берём top-n
    top_movies = sorted_movies.head(n_recommendations)
    
    # Добавляем названия фильмов
    result = []
    for movie_id in top_movies.index:
        avg_rating = top_movies.loc[movie_id, 'avg_rating']
        rating_count = int(top_movies.loc[movie_id, 'rating_count'])
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        result.append((int(movie_id), float(avg_rating), rating_count, title))
    
    return result

def evaluate_rec_systems(
    user_id: int = 610, n_recommendations: int = 10, random_state: int = 42
) -> dict:
    """
    Оценивает эффективность базовых рекомендательных систем.
    """
    # Загружаем данные
    ratings_df, _ = load_data()
    
    # Получаем исторические фильмы пользователя (которые он уже оценил)
    user_history = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    user_history_set = set(user_history)
    
    # 1. Случайные рекомендации
    random_recs = random_recommend(n_recommendations, seed=random_state)
    random_recs_set = set(random_recs)
    
    # 2. Популярные рекомендации
    popular_recs_list = top_n_recommend(n_recommendations, min_ratings=10)
    popular_recs_set = {rec[0] for rec in popular_recs_list}  # Берём movieId из кортежа
    
    # Считаем Accuracy (доля рекомендованных фильмов, которые пользователь уже смотрел)
    random_accuracy = len(random_recs_set & user_history_set) / len(random_recs_set) if len(random_recs_set) > 0 else 0
    popular_accuracy = len(popular_recs_set & user_history_set) / len(popular_recs_set) if len(popular_recs_set) > 0 else 0
    
    return {
        'random_accuracy': random_accuracy,
        'popular_accuracy': popular_accuracy
    }

if __name__ == "__main__":
    # 1. Случайные рекомендации
    print("\n1. СЛУЧАЙНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    random_recs = random_recommend(n_recommendations=10)
    print(f"Рекомендованные ID фильмов: {random_recs}")

    # 2. Популярные фильмы
    print("\n2. ПОПУЛЯРНЫЕ ФИЛЬМЫ (рекомендации на основе популярности):")
    print("-" * 60)
    popular_recs = top_n_recommend(n_recommendations=10)
    print(
        f"{'Rank':<5} {'ID':<6} {'Ср рейтинг':<18} {'Кол-во оценок':<15} {'Название'}"
    )
    print("-" * 60)
    for i, (movie_id, avg_rating, rating_count, title) in enumerate(popular_recs, 1):
        print(
            f"{i:<5} {movie_id:<6} {avg_rating:<18.2f} {rating_count:<15} {title[:50]}"
        )

    # 3. Оценка системы
    print("\n3. ОЦЕНКА КАЧЕСТВА СИСТЕМЫ:")
    print("-" * 60)
    metrics = evaluate_rec_systems()
    print(f"Accuracy (случайные рекомендации): {metrics['random_accuracy']:.4f}")
    print(f"Accuracy (популярные фильмы): {metrics['popular_accuracy']:.4f}")
