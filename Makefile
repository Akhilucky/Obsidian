install:
	 pip install -r requirements.txt
	 docker-compose up -d

cpp:
	 bash cpp/build.sh
	 javac -d java/build java/*.java

java: cpp

test:
	 pytest tests/ -v

dashboard:
	 streamlit run dashboard/app_streamlit.py

backtest:
	 python research/backtester.py

analytics:
	 python analytics/risk.py

execution:
	 python execution/omega.py

clean:
	 docker-compose down -v
