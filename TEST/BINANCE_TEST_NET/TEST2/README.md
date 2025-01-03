# Estratégia de Trading (DCA + Modelo de Machine Learning) com Verificação de SL/TP em Intervalo Diferente

Este projeto demonstra como implementar um robô de trading que:

1. **Opera** seguindo uma **estratégia de DCA** (Dollar-Cost Averaging) com base em sinais fornecidos por um **modelo de Machine Learning**.
2. **Verifica Stop Loss (SL) e Take Profit (TP)** em um **intervalo de tempo diferente** (mais frequente) do que o intervalo principal de candles.

A ideia principal é ter **duas rotinas** rodando em **paralelo**:
- **Rotina A** (principal) roda conforme o fechamento de candle definido em `KLINE_INTERVAL` (por exemplo, 1 hora). Ao fechar cada candle, o robô:
  - Recolhe o candle novo.
  - Recalcula indicadores.
  - (Opcional) Re-treina o modelo de ML a cada `RETRAIN_INTERVAL`.
  - Lê o **sinal** do modelo (compra ou venda).
  - Faz **DCA** (compra ou venda de 70% do saldo) conforme o sinal e se já existe posição aberta ou não.

- **Rotina B** roda **a cada 25 minutos** (ou intervalo que você escolher) para checar **se o preço atingiu o SL ou o TP**. Se for atingido, ela **fecha a posição** (sem precisar esperar o candle fechar).

## Estrutura do Código

No código, temos:

1. **Imports e Configurações**  
   - Carregamento de API Keys, configuração do Client Binance (testnet ou mainnet), e logging.

2. **Variáveis Globais**  
   - `open_position`: booleano que indica se há posição aberta.  
   - `entry_side`: `'buy'` ou `'sell'`.  
   - `entry_price`: guarda o preço de entrada.  
   - `entry_quantity`: quanto foi comprado ou vendido.

3. **Funções Auxiliares**  
   - `get_asset_balance(asset)`: retorna saldo disponível de um ativo.  
   - `get_symbol_price(symbol)`: retorna preço atual (ticker).  
   - `adjust_quantity(...)`: ajusta o tamanho de uma ordem para respeitar os limites do Binance (minQty, stepSize etc.).  
   - `buy_market(...)` e `sell_market(...)`: executam ordens a mercado.  

4. **Stop Loss e Take Profit**  
   - **Stop Loss (SL)**: fecha posição se o preço contra a posição se mover X%.  
     - Se a posição for `'buy'`, SL dispara se `current_price <= entry_price * (1 - STOP_LOSS_PCT)`.  
     - Se a posição for `'sell'`, SL dispara se `current_price >= entry_price * (1 + STOP_LOSS_PCT)`.  
   - **Take Profit (TP)**: fecha posição se o preço a favor atingir Y%.  
     - Se for `'buy'`, TP dispara se `current_price >= entry_price * (1 + TAKE_PROFIT_PCT)`.  
     - Se for `'sell'`, TP dispara se `current_price <= entry_price * (1 - TAKE_PROFIT_PCT)`.  

5. **Checagem de SL/TP em Intervalo Fixo**  
   - Função `check_sl_tp()` faz a **verificação** se SL ou TP foi atingido, e fecha a posição se sim.  
   - Ela **não depende** de candle. Pode ser chamada a qualquer instante.

6. **Modelo de Machine Learning**  
   - É treinado inicialmente ao iniciar o código.  
   - A cada `RETRAIN_INTERVAL` candles, o modelo é re-treinado.  
   - O sinal (`+1` para compra, `-1` para venda) é calculado comparando a **previsão** do próximo Close (target) com o **Close atual**.

7. **Loop Principal (DCA + Modelo)**  
   - `dca_model_loop()`:  
     - Fica em loop infinito.  
     - Chama `wait_for_next_close()` (aguarda o candle fechar conforme `KLINE_INTERVAL`).  
     - Quando o candle fecha, atualiza `df` com o novo candle, recalcula indicadores, re-treina (se for hora), lê o sinal e faz DCA.  

8. **Loop Secundário (SL/TP)**  
   - `sl_tp_loop()`:  
     - Fica em loop infinito.  
     - A cada **25 minutos** (`SL_TP_CHECK_INTERVAL`), chama `check_sl_tp()`.  

9. **Threads**  
   - No `main` (`if __name__ == "__main__":`), criamos duas threads (uso de `threading.Thread`):  
     - Uma para `dca_model_loop()`.  
     - Uma para `sl_tp_loop()`.  

   Dessa forma, **ambas** rodam em paralelo. A thread do modelo só faz operações de DCA em cada candle fechado; enquanto a thread de SL/TP fica verificando se deve fechar a posição a cada 25 minutos.

## Por que Duas Threads?

- **Thread DCA** (intervalo KLINE):  
  - Faz sentido seguir a lógica do candle (por exemplo, 1 hora). O robô só decide **abrir ou aumentar/diminuir** a posição no fechamento do candle (ou seja, cada hora).  
  - Isso evita “ruídos” no intraperíodo e mantém a estratégia ancorada em velas completas.

- **Thread SL/TP** (intervalo fixo menor que o candle):  
  - Se a posição foi aberta e o preço moveu-se rapidamente a favor (TP) ou contra (SL), o trader pode querer sair antes de esperar 1 hora (ou o tempo do candle) para agir.  
  - Ao checar a cada 25 minutos, por exemplo, se o mercado despencar antes do candle acabar, você não segura a posição até o candle fechar.  
  - Assim, o SL/TP é uma espécie de “seguro” que roda em paralelo.

## Fluxo Simplificado

1. **Inicialização**  
   - Carrega dados históricos.  
   - Calcula indicadores.  
   - Treina o modelo (RandomForestRegressor, etc.).  
   - Define variáveis globais (`open_position = False` etc.).  

2. **Thread 1: `dca_model_loop()`**  
   - Loop infinito:  
     - `wait_for_next_close()` => fica esperando 1 hora (ou o que for `KLINE_INTERVAL`).  
     - Ao fechar candle: atualiza DataFrame `df`, recalcula indicadores, se for a hora (`RETRAIN_INTERVAL`), re-treina.  
     - Lê `get_signal(df, model)`.  
     - Se `open_position == False`, abre uma posição (compra ou venda de 70% do saldo).  
     - Se `open_position == True`, faz DCA ou inverte posição, conforme a lógica e o sinal.  

3. **Thread 2: `sl_tp_loop()`**  
   - Loop infinito:  
     - Verifica se há posição (`open_position == True`). Se sim, checa se o **preço atual** atingiu SL ou TP.  
     - Se atingiu, fecha a posição (vende ou compra para zerar).  
     - Dorme por 25 minutos.  

4. **Conclusão**  
   - Com esse design, o robô **não** aguarda o próximo candle para sair em SL/TP. Ele verifica várias vezes durante o candle (a cada 25 minutos), reduzindo possíveis perdas ou garantindo lucros antes do candle seguinte.

## Observações e Boas Práticas

1. **Sincronização de Variáveis**  
   - Como duas threads acessam e modificam `open_position`, `entry_side`, etc., podem ocorrer “race conditions” (conflitos de escrita/leitura). Em códigos reais, use um `threading.Lock` ou outro mecanismo de sincronização para evitar problemas.  
   
2. **Spot vs. Short**  
   - O exemplo assume que podemos “vender a descoberto” (short) no Spot. Na Binance, isso só é possível em **Margin** ou **Futuros**. Se estiver usando Spot comum, vender (`sell`) só funciona se você **tiver** BNB em saldo.  

3. **Parâmetros**  
   - `STOP_LOSS_PCT` e `TAKE_PROFIT_PCT` são fixos no exemplo. Você pode torná-los **dinâmicos** conforme volatilidade ou “euforia” de mercado (ex.: Fear & Greed, ATR etc.).  
   - `RISK_PERCENT_CAPITAL = 0.70` define a fração (70%) do seu saldo que será usada a cada DCA.

4. **Teste e Ajuste**  
   - Faça **backtests** com dados históricos para calibrar os parâmetros (SL, TP, DCA etc.).  
   - Faça **forward tests** (paper trading, testnet) antes de arriscar capital real.

5. **Escalonamento de Tempo**  
   - `KLINE_INTERVAL`: pode ser 1 hora, 4 horas, 15 minutos, etc. Defina conforme sua estratégia.  
   - `SL_TP_CHECK_INTERVAL`: a cada quantos minutos (ou segundos) a thread de SL/TP rodará? Se quiser mais frequente, pode colocar 5 minutos, ou até 1 minuto.

## Conclusão

Este projeto exemplifica como ter **duas rotinas** para um robô de trading:
- **Rotina A** (principal): segue o **candle** e faz **DCA** com base em um **modelo de ML**.  
- **Rotina B** (secundária): verifica **Stop Loss** e **Take Profit** em intervalos menores, para fechar a posição sem aguardar o candle terminar.

Dessa forma, sua estratégia pode aproveitar tanto a consistência de sinais baseados em candles (evitando ruído intraperíodo) quanto a segurança de sair do trade mais cedo se o preço se mover muito contra ou muito a favor.