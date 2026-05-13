
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()
client = TradingClient(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    paper=True
)
account = client.get_account()
print('Connected!')
print(f'Account status : {account.status}')
print(f'Buying power   : \${float(account.buying_power):,.2f}')
