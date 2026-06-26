"""Inspect CryptoHFTDataClient options."""
import cryptohftdata as chd
import inspect
sig = inspect.signature(chd.CryptoHFTDataClient.__init__)
print("CryptoHFTDataClient.__init__ signature:")
print(f"  {sig}")
print()
# Check get_orderbook signature
sig2 = inspect.signature(chd.CryptoHFTDataClient.get_orderbook)
print("get_orderbook signature:")
print(f"  {sig2}")
