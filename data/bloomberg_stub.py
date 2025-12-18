# Bloomberg data stub module
class BloombergData:
    """Bloomberg data fetcher."""
    def bdhs(self, ticker, fields, start_date, end_date):
        """Fetch Bloomberg historical data."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            field: np.random.randn(len(dates)).cumsum() + 100
            for field in fields
        })
        return data

blp = BloombergData()
