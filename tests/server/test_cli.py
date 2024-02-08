import asyncio
import signal
import time
import sys

import pytest


@pytest.fixture(scope="module")
def event_loop_policy(request):
    return asyncio.get_event_loop_policy()  # TODO: windows fixes


@pytest.mark.asyncio
async def test_libertem_server_cli_startup():
    # make sure we can start `libertem-server` and stop it again using ctrl+c
    # this is kind of a smoke test, which should cover the main cli functions.
    p = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'libertem.web.cli', '--no-browser',
        stderr=asyncio.subprocess.PIPE,
    )
    # total deadline, basically how long it takes to import all the dependencies
    # and start the web API
    # (no executor startup is done here)
    deadline = time.monotonic() + 15
    while True:
        if time.monotonic() > deadline:
            assert False, 'timeout'
        line = await asyncio.wait_for(p.stderr.readline(), 5)
        if not line:  # EOF
            assert False, 'subprocess is dead'
        line = line.decode("utf8")
        print('Line:', line, end='')
        if 'LiberTEM listening on' in line:
            break

    async def _debug():
        while True:
            line = await asyncio.wait_for(p.stderr.readline(), 5)
            if not line:  # EOF
                return
            line = line.decode("utf8")
            print('Line@_debug:', line, end='')

    asyncio.ensure_future(_debug())

    try:
        # now, let's kill the subprocess:
        # ctrl+s twice should do the job:
        p.send_signal(signal.SIGINT)
        await asyncio.sleep(0.5)
        if p.returncode is None:
            p.send_signal(signal.SIGINT)

        # wait for the process to stop, but max. 1 second:
        ret = await asyncio.wait_for(p.wait(), 1)
        assert ret == 0
    except Exception:
        if p.returncode is None:
            p.terminate()
            await asyncio.sleep(0.2)
        if p.returncode is None:
            p.kill()
        raise
