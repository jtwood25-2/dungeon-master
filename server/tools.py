"""Claude tools: the ONLY way the DM touches dice or game state. This is what stops the
DM from hallucinating rolls and HP -- every mechanic goes through real, tested code.
"""
import dice as dice_mod
import rules as rules_mod
import ddb as ddb_mod
import encounter as enc_mod
import combat as combat_mod

TOOLS = [
    {"name": "roll_dice",
     "description": "Roll dice for any check, attack, save, or damage. Use real dice for "
                    "everything mechanical instead of inventing numbers. Notation: 1d20+5, "
                    "2d6+3, 4d6kh3, '1d20 adv' (advantage), '1d20 dis' (disadvantage).",
     "input_schema": {"type": "object", "properties": {
         "notation": {"type": "string"},
         "reason": {"type": "string", "description": "what this roll is for"}},
         "required": ["notation"]}},
    {"name": "add_combatant",
     "description": "Register a PC, NPC, or monster so its HP/AC/conditions are tracked.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "max_hp": {"type": "integer"},
         "ac": {"type": "integer"}, "is_pc": {"type": "boolean"},
         "init_mod": {"type": "integer", "description": "initiative modifier"}},
         "required": ["name", "max_hp"]}},
    {"name": "apply_damage",
     "description": "Apply damage to a combatant. HP drops; 0 downs a PC / slays a monster. "
                    "Concentration and death saves are handled automatically — set crit on a "
                    "critical hit, which costs a downed character two death saves.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "amount": {"type": "integer"},
         "crit": {"type": "boolean"}}, "required": ["name", "amount"]}},
    {"name": "heal",
     "description": "Restore HP to a combatant (revives a downed PC).",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "amount": {"type": "integer"}}, "required": ["name", "amount"]}},
    {"name": "set_condition",
     "description": "Add or remove a condition (poisoned, prone, stunned, blessed, ...).",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "condition": {"type": "string"},
         "active": {"type": "boolean", "description": "true=add, false=remove"}},
         "required": ["name", "condition"]}},
    {"name": "start_combat",
     "description": "Roll initiative for the named combatants (or all) and begin combat.",
     "input_schema": {"type": "object", "properties": {
         "names": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "advance_turn",
     "description": "Move to the next combatant's turn (advances the round when it wraps).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "end_combat",
     "description": "End the current combat encounter.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_scene",
     "description": "Update the current scene/location the party is in.",
     "input_schema": {"type": "object", "properties": {
         "scene": {"type": "string"}, "location": {"type": "string"}}, "required": ["scene"]}},
    {"name": "remember",
     "description": "Record a durable world/lore fact so it stays consistent all campaign.",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
    {"name": "lookup_rule",
     "description": "Look up official rules content from the open SRD: a spell, monster stat "
                    "block, or condition. Use this instead of guessing rules or stats.",
     "input_schema": {"type": "object", "properties": {
         "category": {"type": "string", "enum": ["spell", "monster", "condition"]},
         "name": {"type": "string"}}, "required": ["category", "name"]}},
    {"name": "spawn_monster",
     "description": "Look up a monster in the SRD and add N of them to the encounter with real "
                    "HP and AC (e.g. spawn 3 goblins). Then start_combat when ready.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "count": {"type": "integer"}}, "required": ["name"]}},
    {"name": "import_character",
     "description": "Import a player's character sheet from D&D Beyond by its id or URL "
                    "(the sheet must be shared Public). Adds them as a PC with HP/stats.",
     "input_schema": {"type": "object", "properties": {"ddb_id": {"type": "string"}},
                      "required": ["ddb_id"]}},
    {"name": "level_up",
     "description": "Announce a milestone level-up for the WHOLE party. The players level up "
                    "on D&D Beyond themselves; this just marks it pending and tells them to. "
                    "Their numbers do not change until verify_levels confirms it.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "verify_levels",
     "description": "Re-read the players' D&D Beyond sheets and confirm they actually levelled "
                    "up. Call this after they say they're done. Reports who is verified and who "
                    "still shows the old level.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "rest",
     "description": "Take a short or long rest. A long rest restores HP and all spell slots "
                    "and clears death saves; a short rest reports remaining hit dice to spend.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["short", "long"]}}, "required": ["kind"]}},
    {"name": "spend_hit_die",
     "description": "Spend one hit die on a short rest to heal. Pass the die roll the player "
                    "gives you; their CON modifier is added automatically.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "roll": {"type": "integer"}}, "required": ["name", "roll"]}},
    {"name": "death_save",
     "description": "Roll a death saving throw for a downed character. Happens automatically "
                    "at the start of their turn — only call this if a player rolls their own "
                    "(pass their d20) or something forces an extra save.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "roll": {"type": "integer", "description": "their raw d20, if they rolled"}},
         "required": ["name"]}},
    {"name": "concentrate",
     "description": "Mark who is concentrating on which spell. Call this EVERY time a "
                    "concentration spell is cast; the CON save when they take damage is then "
                    "automatic. Pass an empty spell to drop it.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "spell": {"type": "string"}}, "required": ["name", "spell"]}},
    {"name": "inspire",
     "description": "Grant inspiration for good roleplay, a clever plan, or a great line. "
                    "Generous but not constant — roughly once per player per session.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "reason": {"type": "string"}}, "required": ["name"]}},
    {"name": "spend_inspiration",
     "description": "The player spends their inspiration for advantage on a roll.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "loot",
     "description": "Record treasure the party found. They add it to D&D Beyond themselves.",
     "input_schema": {"type": "object", "properties": {
         "items": {"type": "string", "description": "comma-separated"},
         "gold": {"type": "integer"}}}},
    {"name": "downtime",
     "description": "Record what a character did between sessions (training, crafting, "
                    "carousing, research, working a contact). Offer downtime at the START of "
                    "a session after the recap, then pay it off with something concrete.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "activity": {"type": "string"}},
         "required": ["name", "activity"]}},
    {"name": "check_encounter",
     "description": "Weigh a fight against the party BEFORE you run it. Pass the SRD monsters "
                    "you're considering; returns trivial/easy/medium/hard/deadly/LETHAL with "
                    "the maths. Advisory only — use it so a set piece is deadly on purpose "
                    "rather than by accident. Omit monsters to weigh what's already on the board.",
     "input_schema": {"type": "object", "properties": {
         "monsters": {"type": "array", "items": {"type": "object", "properties": {
             "name": {"type": "string"}, "count": {"type": "integer"}}}}}}},
    {"name": "sfx",
     "description": "Play a one-shot sound effect on the players' device at a dramatic beat: "
                    "a blade hit, a door, thunder, a roar. Use it sparingly — a few times a "
                    "scene, on the moments that matter, never on every attack.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "one of the known effect names"}},
         "required": ["name"]}},
    {"name": "add_player",
     "description": "Record who is playing which character (the party roster).",
     "input_schema": {"type": "object", "properties": {
         "player": {"type": "string"}, "character": {"type": "string"}},
         "required": ["player", "character"]}},
]


# Rolls and damage that mean a player has taken their turn. The order used to advance only
# when the model emitted [NEXTTURN], and it frequently narrated the handover in prose instead
# and emitted nothing -- so the turn never moved and the monsters never acted.
_PC_ACTS = {"roll_dice", "apply_damage", "heal", "set_condition", "concentrate",
            "spend_inspiration", "set_resource"}


def _mark_pc_acted(name: str, state) -> None:
    if name not in _PC_ACTS or not getattr(state, "in_combat", False):
        return
    who = state.current_turn()
    c = state.get(who) if who else None
    if c is not None and c.is_pc:
        state.pc_acted = True


def dispatch(name: str, args: dict, state) -> str:
    """Execute a tool call against the game state; return a short text result for the DM."""
    _mark_pc_acted(name, state)
    try:
        if name == "roll_dice":
            r = dice_mod.roll(args["notation"], state.rng)
            why = f" ({args['reason']})" if args.get("reason") else ""
            return f"🎲 {r.notation}{why} = {r.total}  [{r.detail}]"
        if name == "add_combatant":
            c = state.add(args["name"], max_hp=args["max_hp"], ac=args.get("ac", 10),
                          is_pc=args.get("is_pc", False), init_mod=args.get("init_mod", 0))
            return f"Added {c.name}: {c.status}, AC {c.ac}."
        if name == "apply_damage":
            return state.damage(args["name"], args["amount"], args.get("crit", False))
        if name == "heal":
            return state.heal(args["name"], args["amount"])
        if name == "set_condition":
            return state.condition(args["name"], args["condition"], add=args.get("active", True))
        if name == "start_combat":
            # A fight needs something to fight. Starting one with an empty board takes the
            # party into initiative against nobody: no stat blocks, no turns to resolve, and
            # the sniffer that would normally catch a phantom fight goes quiet because
            # in_combat is now true. A playtest spent a whole session in initiative against
            # two goblins that only ever existed in the prose.
            if not any(c for c in state.combatants.values() if not c.is_pc and c.hp > 0):
                return ("[NOT STARTED — there are no enemies on the board.] Run "
                        "[SPAWN <creature> <count>] first, with a real SRD creature name so "
                        "they arrive with a stat block, and then [COMBAT] again. Do not ask "
                        "anyone for initiative until that is done.")
            if not state.map_grid:
                # a fight without a map is the one place people always want one, so don't
                # rely on the DM remembering to ask for it
                state.map_wanted = state.location or state.scene or "the battlefield"
            out = state.start_combat(args.get("names"))
            # Weigh the fight WITHOUT being asked. [CHECK] existed and was never called, so a
            # level-1 character walked into three 22hp creatures -- 2.2x the table's lethal
            # ceiling -- and the first hit took them from full health to dying. The DM cannot
            # be relied on to run an optional safety check in the middle of a scene.
            warn = enc_mod.for_state(state)
            if any(w in warn for w in ("LETHAL", "ABOVE THE TABLE'S CEILING")):
                out += ("\n[BALANCE] " + warn + " Cut numbers or give them an out BEFORE the "
                        "first round -- do not just run it.")
            return out
        if name == "advance_turn":
            # Resolve every monster turn here rather than asking the model to drive them one
            # bracket action at a time. In session 9 the DM narrated a grab, never rolled it,
            # never applied damage and never advanced -- so the player sat at turn_idx 0 for
            # the rest of the fight and never got a go.
            r = combat_mod.run_until_player(state, advance_first=True)
            lines = [e.get("line") or e.get("result", "") for e in r["events"]]
            if who := r.get("player_turn"):
                lines.append(f"Round {r['round']}: {who}'s turn ({r['status']}). "
                             f"Say '{who}, you're up.' and STOP -- do not act for them.")
            elif down := r.get("down"):
                lines.append(
                    f"{down} IS DOWN. Narrate the hit that dropped them and STOP." +
                    ("" if r.get("party_standing") else
                     " Nobody is left standing -- they are dying and there is no one to help. "
                     "Play the consequence (captured, dragged off, found by someone) rather "
                     "than rolling death saves until they die."))
            elif r.get("waiting"):
                lines.append(r["waiting"])
            return "\n".join(x for x in lines if x)
        if name == "end_combat":
            state.director_event = ("combat_end", "")
            return state.end_combat()
        if name == "set_scene":
            state.set_scene(args["scene"], args.get("location"))
            state.director_event = ("scene_change", args.get("location") or args["scene"])
            return "Scene updated."
        if name == "remember":
            state.remember(args["note"])
            return "Noted."
        if name == "lookup_rule":
            r = rules_mod.lookup(args["category"], args["name"])
            if not r:
                return f"No SRD entry for {args['category']} '{args['name']}' (may be non-SRD/copyrighted)."
            if r["kind"] == "monster":
                return f"{r['name']}: HP {r['hp']}, AC {r['ac']}, CR {r['cr']} ({r.get('source')})"
            if r["kind"] == "spell":
                return (f"{r['name']} (lvl {r['level']} {r['school']}): {r['casting_time']}, "
                        f"range {r['range']}, {r['duration']}. {r['desc']}")
            return f"{r['name']}: {r['desc']}"
        if name == "rename":
            return state.rename(args["old"], args["new"])
        if name == "spawn_monster":
            m = rules_mod.monster(args["name"])
            if not m or not m.get("hp"):
                return f"No SRD monster '{args['name']}'."
            count = max(1, int(args.get("count", 1)))
            added = []
            stats = m.get("stats") or {}
            # initiative used to be hardcoded +2 for every creature alike; DEX is right there
            init = ((int(stats.get("dex", 10)) - 10) // 2) if stats.get("dex") else 2
            for i in range(count):
                nm = m["name"] if count == 1 else f"{m['name']} {chr(65 + i)}"
                c = state.add(nm, max_hp=m["hp"], ac=m["ac"] or 10, is_pc=False, init_mod=init,
                              stats=stats)
                # carry the real attacks through. Without these the creature has nothing to
                # roll on its turn and the DM ends up inventing the numbers.
                c.actions = m.get("actions") or []
                c.profile = dict(c.profile or {}, cr=m.get("cr"), type=m.get("type"),
                                 speed=m.get("speed"))
                added.append(c.name)
            atk = len(m.get("actions") or [])
            return (f"Spawned {', '.join(added)} (HP {m['hp']}, AC {m['ac']}, "
                    f"init {init:+d}, {atk} actions each).")
        if name == "import_character":
            return ddb_mod.import_to_state(args["ddb_id"], state)
        if name == "add_player":
            return state.add_player(args["player"], args["character"])
        if name == "set_resource":
            return state.set_resource(args["name"], args["key"], args["value"], args.get("absolute", False))
        if name == "ambience":
            state.ambience = (args.get("mood") or "").strip()
            return f"(ambience → {state.ambience})"
        if name == "level_up":
            return state.level_up()
        if name == "verify_levels":
            return state.verify_levels()
        if name == "rest":
            return state.rest(args.get("kind", "long"))
        if name == "spend_hit_die":
            return state.spend_hit_die(args["name"], args["roll"])
        if name == "death_save":
            return state.death_save(args["name"], args.get("roll"))
        if name == "concentrate":
            return state.concentrate(args["name"], args.get("spell", ""))
        if name == "inspire":
            return state.inspire(args["name"], args.get("reason", ""))
        if name == "spend_inspiration":
            return state.spend_inspiration(args["name"])
        if name == "loot":
            return state.loot(args.get("items", ""), args.get("gold", 0))
        if name == "downtime":
            return state.add_downtime(args["name"], args["activity"])
        if name == "check_encounter":
            return enc_mod.for_state(state, args.get("monsters"))
        if name == "sfx":
            return state.play_sfx(args["name"])
        if name == "set_initiative":
            out = state.set_initiative(args["name"], args["value"])
            # The order is settled. If monsters lead it, resolve them NOW rather than leaving
            # them pending until somebody happens to speak -- with five players and four
            # bandits the monsters lead about half the time, and a player asking "what was the
            # damage?" was answered with two bandit attacks that had been owed since the order
            # was built.
            if state.in_combat and state.order and not state.pending_init:
                lead = combat_mod.brief(state, "")
                if lead:
                    out += "\n" + lead
            return out
        if name == "monster_variants":
            vs = rules_mod.variants(args["name"])
            if not vs:
                return f"No open-content stat block for {args['name']}."
            lines = [f"{len(vs)} version(s) of {args['name']}:"]
            for v in vs:
                lines.append(f"  [{v['source']}] CR {v.get('cr', '?')} AC {v.get('ac', '?')} "
                             f"HP {v.get('hp', '?')} — {(v.get('text') or '')[:110]}")
            return "\n".join(lines)
        if name == "move_token":
            c = state.get(args["name"])
            if not c:
                return f"No {args['name']} here."
            x = ord(str(args["col"]).upper()) - 65
            y = int(args["row"]) - 1
            c.pos = [x, y]
            state.map_dirty = True
            return f"({c.name} → {str(args['col']).upper()}{args['row']})"
        if name == "reveal_area":
            raw = str(args.get("area", "")).strip()
            idx = None
            if raw.isdigit():
                idx = int(raw)
            else:                                    # allow the room's name too
                for i, r in enumerate(state.map_rooms):
                    if len(r) > 4 and raw.lower() in str(r[4]).lower():
                        idx = i; break
            if idx is None or not (0 <= idx < len(state.map_rooms)):
                return f"No map area '{raw}'."
            if idx not in state.map_seen:
                state.map_seen.append(idx)
            state.map_dirty = True
            r = state.map_rooms[idx]
            return f"(revealed {r[4] if len(r) > 4 else 'area ' + str(idx)})"
        if name == "draw_map":
            # Drawing takes seconds, so it is queued and delivered to the players' screen when
            # ready -- the DM must keep narrating rather than waiting on it.
            loc = (args.get("location") or state.location or state.scene or "the area").strip()
            state.map_wanted = loc
            return f"(drawing a map of {loc} — it will appear on their screen shortly)"
        return f"Unknown tool {name}."
    except Exception as e:
        return f"Tool {name} error: {e}"
