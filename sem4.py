"""
Семинар 4. Матричная факторизация
Цель: Разработать методы матричной факторизации для рекомендательной системы.
"""

import numpy as np

from utils import build_user_item_matrix, id_to_movie


def singular_value_decomposition(X: np.array, k: int) -> tuple:
    """
    Разложение матрицы рейтингов X на U, S, V (SVD) и возвращение
    первых k компонент.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if k <= 0:
        raise ValueError("k must be positive")

    U_full, S_full, V_full = np.linalg.svd(X, full_matrices=False)
    k_eff = min(k, S_full.shape[0])
    U = U_full[:, :k_eff]
    S = S_full[:k_eff]
    V = V_full[:k_eff, :]
    return U, S, V


class SVDRecommender:
    """
    Класс для построения рекомендаций на основе матричной факторизации.
    """

    def __init__(self):
        self.ui_matrix = build_user_item_matrix()
        self.U = None
        self.S = None
        self.V = None
        self._build_factorization()

    def _build_factorization(self):
        max_rank = min(self.ui_matrix.shape)
        self.U, self.S, self.V = singular_value_decomposition(
            self.ui_matrix, k=max_rank
        )

    def _reconstruct_matrix(self, k: int) -> np.ndarray:
        """Восстанавливает матрицу рейтингов с первыми k компонентами"""
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Берем первые k компонент
        U_k = self.U[:, :k]
        S_k = self.S[:k]
        V_k = self.V[:k, :]
        
        # Восстанавливаем матрицу: X_hat = U_k @ diag(S_k) @ V_k
        X_hat = U_k @ np.diag(S_k) @ V_k
        
        return X_hat

    def predict_rating(self, user_id: int, item_id: int, k: int = 20) -> float:
        """
        Предсказывает рейтинг user_id по фильму item_id методом низкорангового
        приближения SVD.
        """
        # Восстанавливаем матрицу с k компонентами
        X_hat = self._reconstruct_matrix(k)
        
        # Берем предсказание для нужного пользователя и фильма
        predicted = X_hat[user_id, item_id]
        
        # Обрезаем результат в диапазон [0.0, 5.0]
        return np.clip(predicted, 0.0, 5.0)

    def predict_items_for_user(
        self, user_id: int, k: int = 20, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы для пользователя user_id по SVD.
        """
        # Восстанавливаем матрицу с k компонентами
        X_hat = self._reconstruct_matrix(k)
        
        # Прогнозы для пользователя
        user_predictions = X_hat[user_id]
        
        # Находим фильмы, которые пользователь уже оценил
        user_rated = self.ui_matrix[user_id] > 0
        
        # Исключаем оцененные фильмы
        candidates = np.where(~user_rated)[0]
        
        if len(candidates) == 0:
            return []
        
        # Берем прогнозы для кандидатов
        candidate_scores = user_predictions[candidates]
        
        # Сортируем по убыванию
        sorted_indices = np.argsort(candidate_scores)[::-1]
        top_candidates = candidates[sorted_indices[:n_recommendations]]
        
        return top_candidates.tolist()


if __name__ == "__main__":
    recommender = SVDRecommender()
    k = 100
    U, S, V = singular_value_decomposition(recommender.ui_matrix, k=k)
    print("SVD shapes:", U.shape, S.shape, V.shape)

    X_hat = U @ np.diag(S) @ V
    diff = X_hat - recommender.ui_matrix
    d_min = np.min(diff)
    d_max = np.max(diff)
    d_norm = np.abs(diff).mean()
    print(
        f"Reconstruction error (k={k}): min={d_min:.4f}, max={d_max:.4f}, mean={d_norm:.4f}"
    )

    recs = recommender.predict_items_for_user(1, k=10, n_recommendations=5)
    for rec in recs:
        pred_rating = recommender.predict_rating(1, rec, k=10)
        movie = id_to_movie(rec)
        print(f"Recommend {movie}, predicted rating: {pred_rating:.2f}")