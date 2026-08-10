"""The runner. Knows nothing about ngio, and nothing about any peer library.

Everything here must stay importable inside a child environment holding a
single implementation under test, so the imports are stdlib plus numpy and
nothing else.
"""
