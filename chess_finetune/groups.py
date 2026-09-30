"""Shared identities for leakage checks, splitting, and cluster resampling."""
from urllib.parse import urlparse


def game_identity(row):
    value = row.get('game_id') or row.get('game_url') or row.get('GameUrl')
    if not value:
        return None
    value = str(value).strip()
    if '://' in value:
        parsed = urlparse(value)
        parts = parsed.path.strip('/').split('/')
        if parsed.hostname in ('lichess.org', 'www.lichess.org') and parts:
            # Lichess player links append a four-character player ID to the game ID.
            if len(parts[0]) in (8, 12) and parts[0].isalnum():
                return parts[0][:8]
        return parsed._replace(fragment='', query='').geturl().rstrip('/')
    return value


def identities(row):
    keys = []
    game = game_identity(row)
    if game:
        keys.append(('game', game))
    if row.get('puzzle_id'):
        keys.append(('puzzle', str(row['puzzle_id'])))
    position = row.get('position') or ' '.join(row.get('fen', '').split()[:4])
    if position:
        keys.append(('position', position))
    return keys


def clusters(rows):
    """Connected components: a shared game, puzzle, or position joins two rows."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in identities(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    result = {}
    for i in range(len(rows)):
        result.setdefault(find(i), []).append(i)
    return list(result.values())
