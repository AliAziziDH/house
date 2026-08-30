FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas streamlit plotly pyomo

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "src/portfolio_dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
