# 🚀 Sistema de Predição de Matriz V-Cov com RNN + Insights Gemini AI

Sistema avançado para previsão de matrizes de variância-covariância usando redes neurais LSTM, análise de ponderação baseada em alfa e insights inteligentes com Google AI Studio (Gemini).

## 🎯 Funcionalidades Principais

### 📈 Predição de Matriz V-Cov
- **Redes LSTM**: Modelagem temporal de correlações e volatilidades
- **Decomposição de Cholesky**: Garantia de matrizes positivas definidas
- **Dados em tempo real**: Integração com Yahoo Finance API
- **Visualizações interativas**: Heatmaps de correlação e V-Cov

### 🎯 Ponderação Alfa
- **Cálculo de Alfa/Beta**: Análise CAPM para cada ativo
- **Otimização automática**: Pesos baseados na capacidade de geração de alfa
- **Análise de performance**: Identificação de ativos outperformers
- **Visualizações avançadas**: Gráficos de pesos e dispersão alfa-beta

### 🧠 Insights Gemini AI ⭐ **NOVO!**
- **💬 Comentário de Mercado**: Análises inteligentes sobre ativos selecionados
- **🔮 Insights V-Cov**: Interpretação profissional de matrizes de covariância
- **🎯 Insights Alfa**: Análise especializada de performance de portfólio
- **🤖 Modelos Disponíveis**: Lista e informações dos modelos Gemini disponíveis

## 🏗️ Arquitetura do Sistema

```
projeto/
├── main.py                  # 🚀 Ponto de entrada da aplicação
├── vcov_predictor.py        # 🧮 Core ML: LSTM + V-Cov + Alfa + Gemini
├── gradio_interface.py      # 🌐 Interface web com Gradio
├── alpha_weighting.py       # 🎯 Módulo de ponderação alfa
├── gemini_insights.py       # 🧠 Integração Google AI Studio
├── requirements.txt         # 📦 Dependências do projeto
├── .gitignore              # 🚫 Arquivos ignorados pelo Git
├── GEMINI_CONFIG.md        # 🔑 Configuração Gemini AI
└── README.md               # 📖 Documentação
```

### 🔧 Componentes Principais

#### `vcov_predictor.py` - Motor de Machine Learning
- **VCovPredictor**: Classe principal para predições V-Cov
- **Métodos LSTM**: Modelagem temporal de volatilidades
- **Decomposição de Cholesky**: Garantia matemática de consistência
- **Integração Alfa**: Cálculos de ponderação baseados em performance
- **Integração Gemini**: Métodos para análise inteligente com IA

#### `alpha_weighting.py` - Sistema de Ponderação Alfa
- **AlphaWeighting**: Classe para análise de alfa/beta
- **Cálculo CAPM**: Regressão linear para determinar alfa e beta
- **Otimização de pesos**: Distribuição baseada na capacidade de alfa
- **Análise estatística**: Métricas de performance do portfólio

#### `gemini_insights.py` - Sistema de IA ⭐
- **GeminiInsights**: Classe principal para insights de IA
- **Análise V-Cov**: Interpretação inteligente de matrizes
- **Análise Alfa**: Insights sobre performance de portfólio
- **Comentários de Mercado**: Análises contextuais automatizadas
- **Listagem de Modelos**: Informações sobre modelos disponíveis

#### `gradio_interface.py` - Interface Web
- **Interface tripla**: Abas para V-Cov, Ponderação Alfa e Insights Gemini
- **Visualizações avançadas**: Plotly para gráficos interativos
- **UX otimizada**: Design responsivo e intuitivo
- **Feedback em tempo real**: Progress bars e status updates
- **Integração IA**: Interface completa para funcionalidades Gemini

## 📁 Estrutura do Projeto

```
📦 Projeto/
├── 🐍 main.py                 # Ponto de entrada principal
├── 🧠 vcov_predictor.py       # Lógica de machine learning e processamento
├── 🎨 gradio_interface.py     # Interface gráfica web (Gradio)
├── 📁 venv/                   # Ambiente virtual Python
└── 📄 README.md               # Este arquivo
```

## 🏗️ Arquitetura Modular

### 📄 `main.py`
- **Função**: Ponto de entrada da aplicação
- **Responsabilidade**: Inicializar e lançar a interface Gradio
- **Tamanho**: ~40 linhas (clean & minimal)

### 🧠 `vcov_predictor.py`
- **Função**: Motor de machine learning
- **Responsabilidades**:
  - Download de dados via Yahoo Finance
  - Cálculo de matrizes de variância-covariância
  - Decomposição de Cholesky
  - Treinamento de modelo LSTM
  - Previsões quantitativas
- **Tamanho**: ~300 linhas (core business logic)

### 🎨 `gradio_interface.py`
- **Função**: Interface gráfica web
- **Responsabilidades**:
  - Criação da interface Gradio
  - Visualizações com Plotly
  - Gerenciamento de eventos UI
  - Formatação de resultados
- **Tamanho**: ~200 linhas (presentation layer)

## 🚀 Como Executar

### 1. Preparação do Ambiente
```bash
# Clonar o repositório
git clone https://github.com/brieueu/AI.QUIZZ.git
cd AI.QUIZZ

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar a Aplicação
```bash
# Executar o programa principal
python main.py

# Ou executar diretamente com o Python do venv
"./venv/bin/python" main.py  # Linux/Mac
# ou ".\venv\Scripts\python.exe" main.py  # Windows
```

### 3. Acessar Interface Web
- Abra seu navegador em: `http://localhost:7862`
- Use as abas **Predição V-Cov** e **Ponderação Alfa**

### 4. Exemplos de Uso

#### Predição V-Cov
```
Tickers: AAPL, GOOGL, MSFT, AMZN
Período: 5 anos
Janela: 90 dias
```

#### Ponderação Alfa
```
Tickers: AAPL, GOOGL, MSFT, AMZN, TSLA
Benchmark: ^GSPC (S&P 500)
Taxa livre de risco: 2.0%
```

## 📊 Funcionalidades Detalhadas

### 🔮 Sistema de Predição V-Cov
1. **Download automático** de dados históricos via Yahoo Finance
2. **Cálculo de matrizes** V-Cov históricas com janela deslizante
3. **Decomposição de Cholesky** para manter consistência matemática
4. **Modelagem LSTM** para capturar padrões temporais
5. **Predição futura** da estrutura de correlação
6. **Visualizações interativas** com heatmaps e métricas

### 🎯 Sistema de Ponderação Alfa
1. **Análise CAPM** para cada ativo individualmente
2. **Cálculo de Alfa/Beta** usando regressão linear
3. **Otimização de pesos** baseada na capacidade de geração de alfa
4. **Identificação de outperformers** e underperformers
5. **Visualizações analíticas** com gráficos de barras e dispersão
6. **Recomendações estratégicas** para construção de portfólio

### 📈 Métricas e Análises
- **Volatilidades anualizadas** por ativo
- **Matrizes de correlação** com códigos de cores
- **Análise de concentração** do portfólio
- **Estatísticas de risco-retorno** detalhadas
- **Comparação com benchmark** (S&P 500 padrão)

## 🌐 Acesso

- **URL Local**: http://localhost:7860
- **Interface**: Web-based via Gradio
- **Compatibilidade**: Todos os navegadores modernos

## 🔧 Vantagens da Modularização

### ✅ **Separação de Responsabilidades**
- **Business Logic** isolada da **UI**
- **Facilita testes** unitários
- **Reutilização** de código

### ✅ **Manutenibilidade**
- **Código limpo** e organizado
- **Fácil debugging** e desenvolvimento
- **Extensibilidade** para novas features

### ✅ **Escalabilidade**
- **Diferentes interfaces** (CLI, API, Desktop)
- **Deploy independente** de componentes
- **Performance** otimizada

## 🛠️ Tecnologias Utilizadas

### **Backend (ML)**
- **TensorFlow/Keras**: Redes neurais LSTM
- **NumPy/SciPy**: Computação científica
- **Pandas**: Manipulação de dados
- **scikit-learn**: Pré-processamento
- **yfinance**: Dados financeiros

### **Frontend (UI)**
- **Gradio**: Interface web moderna
- **Plotly**: Visualizações interativas
- **HTML/CSS**: Estilização customizada

## 📊 Funcionalidades

- **📈 Previsão de Matrizes V-Cov**: Usando LSTM + Cholesky
- **🔥 Heatmaps Interativos**: Visualização de correlações
- **📋 Relatórios Detalhados**: Análise quantitativa completa
- **💡 Exemplos Pré-definidos**: Testes rápidos
- **⚡ Processamento em Tempo Real**: Interface responsiva

## 🎯 Casos de Uso

- **Gestão de Portfólio**: Otimização de alocação
- **Análise de Risco**: Medição de volatilidade
- **Trading Quantitativo**: Estratégias baseadas em IA
- **Pesquisa Acadêmica**: Estudos de correlação

## ⚠️ Disclaimer

Este sistema é para fins **educacionais** e de **pesquisa**. Não constitui aconselhamento financeiro.

---
