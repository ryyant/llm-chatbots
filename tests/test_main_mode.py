from unittest.mock import patch


def test_main_default_mode_runs_cli():
    with patch("main.run_cli") as mock_cli, patch("main.run_web") as mock_web, patch(
        "main.load_dotenv"
    ), patch("sys.argv", ["main.py"]):
        from main import main
        main()
    mock_cli.assert_called_once()
    mock_web.assert_not_called()


def test_main_mode_web_runs_web():
    with patch("main.run_cli") as mock_cli, patch("main.run_web") as mock_web, patch(
        "main.load_dotenv"
    ), patch("sys.argv", ["main.py", "--mode", "web"]):
        from main import main
        main()
    mock_web.assert_called_once()
    mock_cli.assert_not_called()


def test_run_web_execs_streamlit():
    with patch("main.os.execvp") as mock_execvp:
        from main import run_web
        run_web()
    args, _ = mock_execvp.call_args
    assert args[0] == "streamlit"
    assert args[1][0] == "streamlit"
    assert args[1][1] == "run"
    assert args[1][2].endswith("app.py")


def test_run_web_handles_missing_streamlit(capsys):
    with patch("main.os.execvp", side_effect=FileNotFoundError), patch(
        "main.sys.exit"
    ) as mock_exit:
        from main import run_web
        run_web()
    out = capsys.readouterr().out
    assert "streamlit" in out.lower()
    mock_exit.assert_called_once_with(1)
