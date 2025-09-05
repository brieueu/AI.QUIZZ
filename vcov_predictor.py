"""
Previsor de Matriz V-Cov com RNN - Módulo de Lógica de Negócio
Engenheiro de Software Sênior - Mercado Financeiro

Sistema avançado para previsão de matrizes de variância-covariância
usando decomposição de Cholesky e redes neurais LSTM.

Autor: Engenheiro de Software Sênior
Especialização: Python, TensorFlow/Keras, Análise Quantitativa
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.linalg import cholesky
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class VCovPredictor:
    """
    Sistema de previsão de matrizes de variância-covariância usando RNN.

    Implementa metodologia avançada com decomposição de Cholesky e LSTM
    para previsão de estruturas de correlação em mercados financeiros.
    """

    def __init__(self):
        """Inicialização do sistema."""
        self.scalers = {}  # Scalers para cada série temporal
        self.model = None

    def predict_vcov_matrix(self, tickers_input, period, window, progress_callback=None):
        """
        Função principal para previsão da matriz V-Cov.
        
        Args:
            tickers_input (str): Tickers separados por vírgula
            period (int): Período histórico em anos
            window (int): Janela V-Cov em dias
            progress_callback (callable): Função callback para atualizações de progresso
        
        Returns:
            dict: Dicionário com resultado_texto, matriz_vcov, matriz_correlacao, tickers
        """
        try:
            # Validação dos inputs
            if not tickers_input.strip():
                raise ValueError("Insira pelo menos um ticker")
            
            if period < 1:
                raise ValueError("Período deve ser maior que 0")
                
            if window < 10:
                raise ValueError("Janela deve ser pelo menos 10 dias")

            # Processar tickers
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            # Etapas do processamento
            steps = [
                "📊 Baixando dados históricos...",
                "🔢 Calculando matrizes V-Cov históricas...",
                "🧮 Aplicando decomposição de Cholesky...",
                "📈 Criando séries temporais...",
                "🤖 Preparando dados para LSTM...",
                "🚀 Treinando modelo LSTM...",
                "🔮 Gerando previsão...",
                "✅ Previsão concluída!"
            ]
            
            # Etapa 1: Download dos dados
            if progress_callback:
                progress_callback(steps[0])
            price_data = self._download_data(tickers, period)
            
            # Etapa 2: Cálculo das matrizes históricas
            if progress_callback:
                progress_callback(steps[1])
            returns = price_data.pct_change().dropna()
            vcov_matrices = self._calculate_rolling_vcov(returns, window)

            # Etapa 3: Decomposição de Cholesky
            if progress_callback:
                progress_callback(steps[2])
            cholesky_matrices = self._apply_cholesky_decomposition(vcov_matrices)

            # Etapa 4: Criação das séries temporais
            if progress_callback:
                progress_callback(steps[3])
            time_series = self._create_time_series(cholesky_matrices)

            # Etapa 5: Preparação dos dados para LSTM
            if progress_callback:
                progress_callback(steps[4])
            X, y = self._prepare_lstm_data(time_series, sequence_length=30)

            # Etapa 6: Treinamento do modelo LSTM
            if progress_callback:
                progress_callback(steps[5])
            self.model = self._build_and_train_lstm(X, y, len(tickers))

            # Etapa 7: Previsão
            if progress_callback:
                progress_callback(steps[6])
            predicted_vcov = self._predict_next_vcov(time_series, len(tickers))

            # Etapa 8: Finalizar
            if progress_callback:
                progress_callback(steps[7])
            
            # Gerar resultado textual
            result_text = self._generate_result_text(predicted_vcov, tickers)
            
            # Calcular matriz de correlação
            std_devs = np.sqrt(np.diag(predicted_vcov))
            correlation_matrix = predicted_vcov / np.outer(std_devs, std_devs)
            
            return {
                'success': True,
                'result_text': result_text,
                'vcov_matrix': predicted_vcov,
                'correlation_matrix': correlation_matrix,
                'tickers': tickers,
                'volatilities': [np.sqrt(predicted_vcov[i, i]) * 100 for i in range(len(tickers))]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result_text': f"❌ Erro durante o processamento: {str(e)}",
                'vcov_matrix': None,
                'correlation_matrix': None,
                'tickers': [],
                'volatilities': []
            }

    def _download_data(self, tickers, period):
        """Download dos dados históricos via yfinance."""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=period)

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )

        # Tratar caso de ticker único
        if len(tickers) == 1:
            prices = data['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Close']

        # Remover dados faltantes
        prices = prices.dropna()

        if prices.empty:
            raise ValueError("Nenhum dado válido encontrado para os tickers fornecidos")

        return prices

    def _calculate_rolling_vcov(self, returns, window):
        """Calcula matrizes de V-Cov em janela deslizante."""
        vcov_matrices = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window:i]
            vcov_matrix = window_returns.cov().values
            vcov_matrices.append(vcov_matrix)

        return np.array(vcov_matrices)

    def _apply_cholesky_decomposition(self, vcov_matrices):
        """Aplica decomposição de Cholesky nas matrizes V-Cov."""
        cholesky_matrices = []

        for vcov_matrix in vcov_matrices:
            try:
                # Adicionar pequena regularização se necessário
                vcov_matrix += np.eye(vcov_matrix.shape[0]) * 1e-6
                L = cholesky(vcov_matrix, lower=True)
                cholesky_matrices.append(L)
            except np.linalg.LinAlgError:
                # Se falhar, usar decomposição SVD como fallback
                U, s, Vt = np.linalg.svd(vcov_matrix)
                s = np.maximum(s, 1e-6)  # Regularização
                L = U @ np.diag(np.sqrt(s))
                cholesky_matrices.append(L)

        return np.array(cholesky_matrices)

    def _create_time_series(self, cholesky_matrices):
        """Extrai elementos da matriz triangular inferior como séries temporais."""
        n = cholesky_matrices.shape[1]  # Número de ativos
        n_series = n * (n + 1) // 2  # Número de séries temporais

        time_series = np.zeros((len(cholesky_matrices), n_series))

        for t, L in enumerate(cholesky_matrices):
            idx = 0
            for i in range(n):
                for j in range(i + 1):  # Apenas elementos da triangular inferior
                    time_series[t, idx] = L[i, j]
                    idx += 1

        return time_series

    def _prepare_lstm_data(self, time_series, sequence_length=30):
        """Prepara dados no formato adequado para LSTM."""
        n_samples, n_features = time_series.shape

        # Normalização das séries temporais
        self.scalers = {}
        normalized_series = np.zeros_like(time_series)

        for i in range(n_features):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_series[:, i] = scaler.fit_transform(
                time_series[:, i].reshape(-1, 1)
            ).flatten()
            self.scalers[i] = scaler

        # Criação das sequências
        X, y = [], []

        for i in range(sequence_length, n_samples):
            X.append(normalized_series[i - sequence_length:i])
            y.append(normalized_series[i])

        return np.array(X), np.array(y)

    def _build_and_train_lstm(self, X, y, n_assets):
        """Constrói e treina o modelo LSTM."""
        n_features = X.shape[2]

        # Arquitetura do modelo
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(n_features, activation='tanh')
        ])

        # Compilação
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Treinamento
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )

        return model

    def _predict_next_vcov(self, time_series, n_assets):
        """Gera previsão da próxima matriz V-Cov."""
        sequence_length = 30

        # Preparar última sequência
        last_sequence = time_series[-sequence_length:]

        # Normalizar usando os scalers treinados
        normalized_sequence = np.zeros_like(last_sequence)
        for i in range(last_sequence.shape[1]):
            normalized_sequence[:, i] = self.scalers[i].transform(
                last_sequence[:, i].reshape(-1, 1)
            ).flatten()

        # Fazer previsão
        X_pred = normalized_sequence.reshape(1, sequence_length, -1)
        y_pred_normalized = self.model.predict(X_pred, verbose=0)[0]

        # Desnormalizar previsão
        y_pred = np.zeros_like(y_pred_normalized)
        for i in range(len(y_pred_normalized)):
            y_pred[i] = self.scalers[i].inverse_transform(
                y_pred_normalized[i].reshape(-1, 1)
            )[0, 0]

        # Reconstruir matriz de Cholesky
        L_pred = np.zeros((n_assets, n_assets))
        idx = 0
        for i in range(n_assets):
            for j in range(i + 1):
                L_pred[i, j] = y_pred[idx]
                idx += 1

        # Reconstruir matriz V-Cov
        vcov_pred = L_pred @ L_pred.T

        # Anualizar (252 dias úteis)
        vcov_pred_annual = vcov_pred * 252

        return vcov_pred_annual

    def _generate_result_text(self, predicted_vcov, tickers):
        """Gera o texto formatado do resultado."""
        result_text = "# 📊 MATRIZ DE VARIÂNCIA-COVARIÂNCIA PREVISTA (ANUALIZADA)\n\n"
        
        # Informações gerais
        result_text += f"**Ativos:** {', '.join(tickers)}\n"
        result_text += f"**Data da previsão:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Matriz formatada
        result_text += "## 📈 Matriz V-Cov (valores em formato científico):\n\n"
        
        # Criar tabela markdown
        result_text += "| Ativo |"
        for ticker in tickers:
            result_text += f" {ticker} |"
        result_text += "\n|-------|"
        for _ in tickers:
            result_text += "------|"
        result_text += "\n"
        
        # Linhas da matriz
        for i, ticker in enumerate(tickers):
            result_text += f"| **{ticker}** |"
            for j in range(len(tickers)):
                result_text += f" {predicted_vcov[i, j]:.4e} |"
            result_text += "\n"

        # Volatilidades (diagonal)
        result_text += "\n## 📊 Volatilidades Anualizadas:\n\n"
        for i, ticker in enumerate(tickers):
            vol = np.sqrt(predicted_vcov[i, i]) * 100
            result_text += f"- **{ticker}**: {vol:.2f}%\n"

        return result_text
