install:
	 pip install -r requirements.txt
	 docker-compose up -d

cpp:
	 bash cpp/build.sh
	 javac -d java/build java/*.java

java: cpp

test:
	 pytest tests/ -v

api:
	 python api/server.py

web:
	 cd web && npm run dev

install:
	 pip install -r requirements.txt
	 cd web && npm install
	 bash cpp/build.sh
	 javac -d java/build java/*.java

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
