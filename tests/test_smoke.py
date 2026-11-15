
def test_the_command_line_runs(monkeypatch, capsys):
    from app import cli

    monkeypatch.setattr(
        cli,
        "load_model",
        lambda: (None, None, {})
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: "4"
    )

    cli.main()

    output = capsys.readouterr().out

    assert "ADVENTURE STORY WEAVER" in output
    assert "Thank you for using Adventure Story Weaver!" in output
