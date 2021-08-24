mkdir -p ~/.streamlit/

echo "[theme]
base="light"
primaryColor="#e4ad86"
secondaryBackgroundColor="#fbe8e8"
textColor="#908263"
font = "sans serif"
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml