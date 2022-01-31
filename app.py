import streamlit as st 
import yfinance as yf 
from datetime import date
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

DATA_INICIO = '2017-01-01'
DATA_FIM = date.today().strftime('%Y-%m-%d')

st.title('Análise de ações')

st.sidebar.header('Escolha a ação')


n_dias = st.slider('Quantidade e dias de previsão', 30, 365)

def pagar_dados_acoes():
    path = 'acoes.csv'
    return pd.read_csv(path, delimiter=';')

df = pagar_dados_acoes()

acao = df['snome']
nome_acao_esolhida = st.sidebar.selectbox('Escolha uma ação: ', acao)

df_acao = df[df['snome'] ==  nome_acao_esolhida]
acao_escolhida = df_acao.iloc[0]['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'

@st.cache
def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, DATA_INICIO, DATA_FIM)
    df.reset_index(inplace=True)
    return df

df_valores = pegar_valores_online(acao_escolhida)

st.subheader('Tabela de valores -' + nome_acao_esolhida)
st.write(df_valores.tail(10))

st.subheader('Gráfico de preços')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Close'],
                         name='Preço Abertura',
                         line_color='yellow'))

fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Open'],
                         name='Preço Abertura',
                         line_color='blue'))        

st.plotly_chart(fig)

df_treino = df_valores[['Date','Close']]

df_treino = df_treino.rename(columns={'Date':'ds','Close':'y'})

modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
previsao = modelo.predict(futuro)

st.subheader('Previsão')
st.write(previsao[['ds','yhat','yhat_lower','yhat_upper']].tail(n_dias))

grafico1 = plot_plotly(modelo, previsao)
st.plotly_chart(grafico1)

grafico2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(grafico2)