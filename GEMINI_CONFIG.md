# Configuração Google AI Studio (Gemini)

## 🔑 Como obter uma API Key

1. **Acesse**: https://aistudio.google.com/app/apikey
2. **Faça login** com sua conta Google
3. **Clique em** "Create API Key"
4. **Copie** a chave gerada (formato: `AIzaSy...`)

## 🚀 Como configurar

### Método 1: Variável de Ambiente (Recomendado)
```bash
export GOOGLE_AI_API_KEY="sua_api_key_real_aqui"
```

### Método 2: Interface Web (Mais Comum)
1. Acesse `http://localhost:7862`
2. Vá para "🧠 Insights Gemini AI"  
3. Digite sua API Key no campo "🔑 API Key Google AI Studio"
4. Use qualquer funcionalidade do Gemini

## 📋 Funcionalidades Disponíveis

### 💬 Comentário de Mercado
- Análises inteligentes sobre ativos selecionados
- Tendências de mercado e setoriais
- Riscos e oportunidades identificados
- Outlook de curto e médio prazo

### 🔮 Insights V-Cov
- Interpretação profissional de matrizes de covariância
- Análise de risco e diversificação
- Identificação de concentrações perigosas
- Recomendações de otimização de portfólio

### 🎯 Insights Alfa
- Análise especializada de performance alfa/beta
- Interpretação de métricas CAPM
- Identificação de geradores de alfa
- Estratégias de investimento recomendadas

### 🤖 Modelos Disponíveis
- Lista completa de modelos Gemini disponíveis
- Informações sobre capacidades e limites
- Recomendações de uso por tipo de análise

## ⚠️ Informações Importantes

- **Modelos Atualizados**: Sistema usa gemini-1.5-pro, gemini-1.5-flash, ou gemini-2.0-flash-exp
- **Modelo Antigo**: gemini-pro foi descontinuado pela Google
- **Gratuito**: Google AI Studio oferece cota gratuita generosa
- **Rate Limits**: Respeite os limites (2 req/min para 1.5-pro, 15 req/min para 1.5-flash)
- **Privacidade**: API key não é armazenada permanentemente
- **Qualidade**: Gemini 1.5 Pro é otimizado para análises complexas (contexto de 2M tokens)

## ⚡ Teste Rápido
```python
from vcov_predictor import VCovPredictor
predictor = VCovPredictor()

# Configurar API key
predictor.configure_gemini("sua_api_key_aqui")

# Testar listagem de modelos
modelos = predictor.list_gemini_models()
print(modelos)

# Testar comentário de mercado
comentario = predictor.get_market_commentary(["AAPL", "GOOGL", "MSFT"])
print(comentario)
```

## 🚫 Integração ChatGPT Removida
- Sistema agora usa apenas Google AI Studio (Gemini)
- Focamos em uma única integração de IA de alta qualidade
- Reduzimos complexidade e melhoramos confiabilidade
- **Atualizado**: Modelos Gemini 1.5/2.0 com capacidades expandidas

## 🔄 Modelos Suportados
- **gemini-1.5-pro**: Análises complexas, contexto 2M tokens
- **gemini-1.5-flash**: Análises rápidas, contexto 1M tokens  
- **gemini-2.0-flash-exp**: Modelo experimental mais recente
- **Fallback automático**: Sistema tenta diferentes modelos se um falhar
