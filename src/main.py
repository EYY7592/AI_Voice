"""ScamLens-TW localhost 啟動入口。"""
import uvicorn


def main() -> None:
    uvicorn.run("src.gui:app", host="127.0.0.1", port=7861, reload=False)


if __name__ == "__main__":
    main()
