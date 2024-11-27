import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    data = pd.read_csv(file_name)
    data['Data'] = pd.to_datetime(data['Data'])
    data = data[['Data', 'Zamkniecie']]
    return data

def calculate_macd(data):
    data['EMA_12'] = data['Zamkniecie'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Zamkniecie'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    return data

def calculate_signal(data):
    data['SIGNAL'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data
def find_crossings(data1, data2):
    buy_signals = []
    sell_signals = []
    for i in range(1, len(data1)):
        if data1[i-1] < data2[i-1] and data1[i] > data2[i]:
            buy_signals.append(i)
        elif data1[i-1] > data2[i-1] and data1[i] < data2[i]:
            sell_signals.append(i)
    return buy_signals, sell_signals

def plot_stock_data(data):
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(data['Data'], data['Zamkniecie'], color='black', label='Wartość')
    plt.title('Notowania analizowanego instrumentu finansowego')
    plt.xlabel('Data')
    plt.ylabel('Cena z zamknięcia')
    plt.legend()
    plt.show()

def plot_macd_signals(data, macd, signal, buy_signals, sell_signals):
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(data['Data'], macd, color='midnightblue', label='MACD')
    plt.plot(data['Data'], signal, color='aqua', label='SIGNAL')
    plt.scatter(data['Data'][buy_signals], [macd[i] for i in buy_signals], color='green', label='Kupno')
    plt.scatter(data['Data'][sell_signals], [macd[i] for i in sell_signals], color='red', label='Sprzedaż')
    plt.title('MACD+SIGNAL z punktami kupna/sprzedaży')
    plt.xlabel('Data')
    plt.ylabel('Wartość')
    plt.legend()
    plt.show()

def plot_stock_with_signals(data, buy_signals, sell_signals):
    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(data['Data'], data['Zamkniecie'], color='black', label='Wartość')
    plt.scatter(data['Data'][buy_signals], [data['Zamkniecie'][i] for i in buy_signals], color='green', label='Kupno')
    plt.scatter(data['Data'][sell_signals], [data['Zamkniecie'][i] for i in sell_signals], color='red', label='Sprzedaż')
    plt.title('Wykres transakcji kupna/sprzedaży')
    plt.xlabel('Data')
    plt.ylabel('Cena z zamknięcia')
    plt.legend()
    plt.show()

def trading_algorithm(data, buy_signals, sell_signals):
    capital = 0.0
    units = 1000.0
    start_capital = data['Zamkniecie'][0] * units
    for i in range(len(data)):
        if i in buy_signals and capital > 0:
            units = capital / data['Zamkniecie'][i]
            capital = 0.0
        elif i in sell_signals and units > 0:
            capital = units * data['Zamkniecie'][i]
            units = 0.0
    if units > 0:
        final_capital = units * data['Zamkniecie'].iloc[-1]
    else:
        final_capital = capital
    return start_capital, final_capital

data = load_data('sp500_6months.csv')

data = calculate_macd(data)
data = calculate_signal(data)

macd = data['MACD']
signal = data['SIGNAL']

buy_signals, sell_signals = find_crossings(macd, signal)

plot_stock_data(data)
plot_macd_signals(data, macd, signal, buy_signals, sell_signals)
plot_stock_with_signals(data, buy_signals, sell_signals)

start_capital, final_capital = trading_algorithm(data, buy_signals, sell_signals)
print(f"Początkowy kapitał: {start_capital}")
print(f"Końcowy kapitał: {final_capital}")