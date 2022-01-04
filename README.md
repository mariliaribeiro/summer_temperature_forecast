# Análise exploratória e modelo de predição de temperatura utilizando redes neurais recorrentes do tipo LSTM

Este notebook tem por objetivo realizar uma análise exploratória da base de dados [Bias correction of numerical prediction model temperature forecast Data Set](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast) disponível no repositório da UCI para criar um modelo de preção de temperaturas máxima e mínima do dia seguinte durante o verão em uma cidade da Coreia do Sul.
A base de dados possui medições de 25 estações metereológicas durante o verão de 2013 a 2017. Ao todo são 25 variáveis para cada estação. Abaixo a descrição de cada variável disponível.
Neste notebook iremos utilizar um modelo de redes neurais recorrentes (RNN) do tipo LSTM pra realizar as predições. As RNNs vem se mostrando muito eficientes como uma alternativa para realizar previsões de séries temporais. A arquitetura LSTM surgiu como uma alternativa para resolver o problema do Vanishing Gradient da versão vanila das RNNs, pois consegue aprender conexões de longo prazo. Saiba mais sobre a arquitetura LSTM no post [Redes Neurais | LSTM](https://medium.com/turing-talks/turing-talks-27-modelos-de-predi%C3%A7%C3%A3o-lstm-df85d87ad210) do Fernando Matsumoto.

**Descrição dos dados disponíveis**


| Campo | Descrição |
| -------- | ------ |
| station | Número da estação meteorológica usada: 1 a 25 |
| Date | Dia atual: aaaa-mm-dd ('2013-06-30' a '2017-08-30') |
| Present_Tmax | Temperatura máxima do ar entre 0 e 21 h do dia atual (Â°C): 20 a 37,6 |
| Present_Tmin | Temperatura mínima do ar entre 0 e 21 h do dia atual (Â°C): 11,3 a 29,9 |
| LDAPS_RHmin | Previsão do modelo LDAPS da umidade relativa mínima no dia seguinte (%): 19,8 a 98,5 |
| LDAPS_RHmax | Previsão do modelo LDAPS da umidade relativa máxima no dia seguinte (%): 58,9 a 100 |
| LDAPS_Tmax_lapse | Previsão do modelo LDAPS da taxa de lapso aplicada da temperatura máxima do ar no dia seguinte (Â°C): 17,6 a 38,5 |
| LDAPS_Tmin_lapse | Previsão do modelo LDAPS da taxa de lapso aplicada da temperatura mínima do ar no dia seguinte (Â°C): 14,3 a 29,6 |
| LDAPS_WS | Previsão do modelo LDAPS da velocidade média do vento no dia seguinte (m / s): 2,9 a 21,9 |
| LDAPS_LH | Previsão do modelo LDAPS do fluxo de calor latente médio do dia seguinte (W / m2): -13,6 a 213,4 |
| LDAPS_CC1 | Previsão do modelo LDAPS da cobertura média de nuvem dividida nas primeiras 6 horas do dia seguinte (0-5 h) ( %): 0 a 0,97 |
| LDAPS_CC2 | Previsão do modelo LDAPS do 2º dia seguinte de 6 horas de cobertura de nuvem média (6-11 h) (%): 0 a 0,97 |
| LDAPS_CC3 | Previsão do modelo LDAPS do 3º dia seguinte para cobertura de nuvens média da divisão de 6 horas (12-17 h) (%): 0 a 0,98 |
| LDAPS_CC4 | Previsão do modelo LDAPS do 4º dia seguinte para cobertura de nuvens média da divisão de 6 horas (18-23 h) (%): 0 a 0,97 |
| LDAPS_PPT1 | Previsão do modelo LDAPS da precipitação média da divisão de 6 horas no dia seguinte (0-5 h) (%): 0 a 23,7 |
| LDAPS_PPT2 | Previsão do modelo LDAPS da precipitação média da divisão de 6 horas no dia seguinte ( 6-11 h) (%): 0 a 21,6 |
| LDAPS_PPT3 | Previsão do modelo LDAPS da precipitação média dividida em 6 horas no dia seguinte (12-17 h) (%): 0 a 15,8 |
| LDAPS_PPT4 | Previsão do modelo LDAPS da 4ª precipitação média dividida em 6 horas no dia seguinte (18-23 h) (%): 0 a 16,7 |
| lat | Latitude (Â°): 37,456 a 37,645 |
| lon | Longitude (Â°): 126,826 a 127,135 |
| DEM | Elevação (m): 12,4 a 212,3 |
| Slope | Slope (Â°): 0,1 a 5,2 |
| Solar radiation | Radiação solar de entrada diária (wh / m2): 4329,5 a 5992,9 |
| Next_Tmax | A temperatura máxima do ar no dia seguinte (Â°C): 17,4 a 38,9 |
| Next_Tmin | A temperatura mínima do ar no dia seguinte (Â°C): 11,3 a 29,8 |