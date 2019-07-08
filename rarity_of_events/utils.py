
def map_actions_3v3(actions):
    mapped_actions = []
    x_positions = []
    y_positions = []

    for act in actions:
        action = act.item()
        mapped_action = 0
        x, y = None, None

        # NON-SPATIAL ACTIONS
        if action == 0:
            mapped_action = 0
        if action == 1:
            mapped_action = 1
        if action == 2:
            mapped_action = 2
        if action == 3:
            mapped_action = 3
        if action == 4:
            mapped_action = 4
        if action == 5:
            mapped_action = 5
        if action == 6:
            mapped_action = 6
        if action == 7:
            mapped_action = 7
        if action == 8:
            mapped_action = 8
        if action == 9:
            mapped_action = 10
        if action == 10:
            mapped_action = 11
        if action == 11:
            mapped_action = 12
        if action == 12:
            mapped_action = 13
        if action == 13:
            mapped_action = 14
        if action == 14:
            mapped_action = 15
        if action == 15:
            mapped_action = 20
        if action == 16:
            mapped_action = 33
        if action == 17:
            mapped_action = 34
        if action == 18:
            mapped_action = 35
        if action == 19:
            mapped_action = 36

        # SPATIAL ACTIONS
        if 20 <= action <= (20+59):  # STAND-UP
            mapped_action = 9
            position = action - 20
            x, y = get_position_field_3v3(position)
        if 80 <= action <= (80+59):  # PLACE PLAYER
            mapped_action = 16
            position = action - 80
            x, y = get_position_field_3v3(position)
        if 140 <= action <= (140+59):  # PLACE BALL
            mapped_action = 17
            position = action - 140
            x, y = get_position_field_3v3(position)
        if 200 <= action <= (200+98):  # PUSH
            mapped_action = 18
            position = action - 200
            x, y = get_position_board_3v3(position)
        if 298 <= action <= (298+59):  # FOLLOW-UP
            mapped_action = 19
            position = action - 298
            x, y = get_position_field_3v3(position)
        if 358 <= action <= (358+59):  # SELECT PLAYER
            mapped_action = 21
            position = action - 358
            x, y = get_position_field_3v3(position)
        if 418 <= action <= (418+59):  # MOVE
            mapped_action = 22
            position = action - 418
            x, y = get_position_field_3v3(position)
        if 478 <= action <= (478+59):  # BLOCK
            mapped_action = 23
            position = action - 478
            x, y = get_position_field_3v3(position)
        if 538 <= action <= (538+59):  # PASS
            mapped_action = 24
            position = action - 538
            x, y = get_position_field_3v3(position)
        if 598 <= action <= (598+59):  # FOUL
            mapped_action = 25
            position = action - 598
            x, y = get_position_field_3v3(position)
        if 658 <= action <= (658+59):  # HANDOFF
            mapped_action = 26
            position = action - 658
            x, y = get_position_field_3v3(position)
        if 718 <= action <= (718+59):  # START MOVE
            mapped_action = 27
            position = action - 718
            x, y = get_position_field_3v3(position)
        if 778 <= action <= (778+59):  # START BLOCK
            mapped_action = 28
            position = action - 778
            x, y = get_position_field_3v3(position)
        if 838 <= action <= (838+59):  # START BLITZ
            mapped_action = 29
            position = action - 838
            x, y = get_position_field_3v3(position)
        if 898 <= action <= (898+59):  # START PASS
            mapped_action = 30
            position = action - 898
            x, y = get_position_field_3v3(position)
        if 958 <= action <= (958+59):  # START FOUL
            mapped_action = 31
            position = action - 958
            x, y = get_position_field_3v3(position)
        if 1018 <= action <= (1018+59):  # START HANDOFF
            mapped_action = 32
            position = action - 1018
            x, y = get_position_field_3v3(position)

        mapped_actions.append(mapped_action)
        x_positions.append(x)
        y_positions.append(y)

    return mapped_actions, x_positions, y_positions


def map_actions_1v1(actions):
    mapped_actions = []
    x_positions = []
    y_positions = []

    for act in actions:
        action = act.item()
        mapped_action = 0
        position = None
        x, y = None, None

        # NON-SPATIAL ACTIONS
        if action == 0:
            mapped_action = 0
        if action == 1:
            mapped_action = 1
        if action == 2:
            mapped_action = 2
        if action == 3:
            mapped_action = 3
        if action == 4:
            mapped_action = 4
        if action == 5:
            mapped_action = 5
        if action == 6:
            mapped_action = 6
        if action == 7:
            mapped_action = 7
        if action == 8:
            mapped_action = 8
        if action == 9:
            mapped_action = 10
        if action == 10:
            mapped_action = 11
        if action == 11:
            mapped_action = 12
        if action == 12:
            mapped_action = 13
        if action == 13:
            mapped_action = 14
        if action == 14:
            mapped_action = 15
        if action == 15:
            mapped_action = 20
        if action == 16:
            mapped_action = 33
        if action == 17:
            mapped_action = 34
        if action == 18:
            mapped_action = 35
        if action == 19:
            mapped_action = 36

        # SPATIAL ACTIONS
        if 20 <= action <= 31:  # STAND-UP
            mapped_action = 9
            position = action - 20
            x, y = get_position_field_1v1(position)
        if 32 <= action <= 43:  # PLACE PLAYER
            mapped_action = 16
            position = action - 32
            x, y = get_position_field_1v1(position)
        if 44 <= action <= 55:  # PLACE BALL
            mapped_action = 17
            position = action - 44
            x, y = get_position_field_1v1(position)
        if 56 <= action <= 85:  # PUSH
            mapped_action = 18
            position = action - 56
            x, y = get_position_board_1v1(position)
        if 86 <= action <= 97:  # FOLLOW-UP
            mapped_action = 19
            position = action - 86
            x, y = get_position_field_1v1(position)
        if 98 <= action <= 109:  # SELECT PLAYER
            mapped_action = 21
            position = action - 98
            x, y = get_position_field_1v1(position)
        if 110 <= action <= 121:  # MOVE
            mapped_action = 22
            position = action - 110
            x, y = get_position_field_1v1(position)
        if 122 <= action <= 133:  # BLOCK
            mapped_action = 23
            position = action - 122
            x, y = get_position_field_1v1(position)
        if 134 <= action <= 145:  # PASS
            mapped_action = 24
            position = action - 134
            x, y = get_position_field_1v1(position)
        if 146 <= action <= 147:  # FOUL
            mapped_action = 25
            position = action - 146
            x, y = get_position_field_1v1(position)
        if 158 <= action <= 169:  # HANDOFF
            mapped_action = 26
            position = action - 158
            x, y = get_position_field_1v1(position)
        if 170 <= action <= 181:  # START MOVE
            mapped_action = 27
            position = action - 170
            x, y = get_position_field_1v1(position)
        if 182 <= action <= 193:  # START BLOCK
            mapped_action = 28
            position = action - 182
            x, y = get_position_field_1v1(position)
        if 194 <= action <= 205:  # START BLITZ
            mapped_action = 29
            position = action - 194
            x, y = get_position_field_1v1(position)
        if 206 <= action <= 217:  # START PASS
            mapped_action = 30
            position = action - 206
            x, y = get_position_field_1v1(position)
        if 218 <= action <= 229:  # START FOUL
            mapped_action = 31
            position = action - 218
            x, y = get_position_field_1v1(position)
        if 230 <= action <= 241:  # START HANDOFF
            mapped_action = 32
            position = action - 230
            x, y = get_position_field_1v1(position)

        mapped_actions.append(mapped_action)
        x_positions.append(x)
        y_positions.append(y)

    return mapped_actions, x_positions, y_positions


def map_actions_fully_conv(actions):
    mapped_actions = []
    x_positions = []
    y_positions = []

    for act in actions:
        action = act.item()
        mapped_action = 0
        position = None
        x, y = None, None

        # NON-SPATIAL ACTIONS
        if action == 0:
            mapped_action = 0
        if action == 1:
            mapped_action = 1
        if action == 2:
            mapped_action = 2
        if action == 3:
            mapped_action = 3
        if action == 4:
            mapped_action = 4
        if action == 5:
            mapped_action = 5
        if action == 6:
            mapped_action = 6
        if action == 7:
            mapped_action = 7
        if action == 8:
            mapped_action = 8
        if action == 9:
            mapped_action = 10
        if action == 10:
            mapped_action = 11
        if action == 11:
            mapped_action = 12
        if action == 12:
            mapped_action = 13
        if action == 13:
            mapped_action = 14
        if action == 14:
            mapped_action = 15
        if action == 15:
            mapped_action = 20
        if action == 16:
            mapped_action = 33
        if action == 17:
            mapped_action = 34
        if action == 18:
            mapped_action = 35
        if action == 19:
            mapped_action = 36

        # SPATIAL ACTIONS
        if 20 <= action <= 49:  # STAND-UP
            mapped_action = 9
            position = action - 20
            x, y = get_position_board_1v1(position)
        if 50 <= action <= 79:  # PLACE PLAYER
            mapped_action = 16
            position = action - 50
            x, y = get_position_board_1v1(position)
        if 80 <= action <= 109:  # PLACE BALL
            mapped_action = 17
            position = action - 80
            x, y = get_position_board_1v1(position)
        if 110 <= action <= 139:  # PUSH
            mapped_action = 18
            position = action - 110
            x, y = get_position_board_1v1(position)
        if 140 <= action <= 169:  # FOLLOW-UP
            mapped_action = 19
            position = action - 140
            x, y = get_position_board_1v1(position)
        if 170 <= action <= 199:  # SELECT PLAYER
            mapped_action = 21
            position = action - 170
            x, y = get_position_board_1v1(position)
        if 200 <= action <= 229:  # MOVE
            mapped_action = 22
            position = action - 200
            x, y = get_position_board_1v1(position)
        if 230 <= action <= 259:  # BLOCK
            mapped_action = 23
            position = action - 230
            x, y = get_position_board_1v1(position)
        if 260 <= action <= 289:  # PASS
            mapped_action = 24
            position = action - 260
            x, y = get_position_board_1v1(position)
        if 290 <= action <= 319:  # FOUL
            mapped_action = 25
            position = action - 290
            x, y = get_position_board_1v1(position)
        if 320 <= action <= 349:  # HANDOFF
            mapped_action = 26
            position = action - 320
            x, y = get_position_board_1v1(position)
        if 350 <= action <= 379:  # START MOVE
            mapped_action = 27
            position = action - 350
            x, y = get_position_board_1v1(position)
        if 380 <= action <= 409:  # START BLOCK
            mapped_action = 28
            position = action - 380
            x, y = get_position_board_1v1(position)
        if 410 <= action <= 439:  # START BLITZ
            mapped_action = 29
            position = action - 410
            x, y = get_position_board_1v1(position)
        if 440 <= action <= 469:  # START PASS
            mapped_action = 30
            position = action - 440
            x, y = get_position_board_1v1(position)
        if 470 <= action <= 499:  # START FOUL
            mapped_action = 31
            position = action - 470
            x, y = get_position_board_1v1(position)
        if 500 <= action <= 429:  # START HANDOFF
            mapped_action = 32
            position = action - 500
            x, y = get_position_board_1v1(position)

        mapped_actions.append(mapped_action)
        x_positions.append(x)
        y_positions.append(y)

    return mapped_actions, x_positions, y_positions


def get_position_field_1v1(pos):

    x, y = None, None

    if pos == 0:
        x, y = 1, 1
    if pos == 1:
        x, y = 2, 1
    if pos == 2:
        x, y = 3, 1
    if pos == 3:
        x, y = 4, 1
    if pos == 4:
        x, y = 1,2
    if pos == 5:
        x, y = 2, 2
    if pos == 6:
        x, y = 3, 2
    if pos == 7:
        x, y = 4, 2
    if pos == 8:
        x, y = 1, 3
    if pos == 9:
        x, y = 2, 3
    if pos == 10:
        x, y = 3, 3
    if pos == 11:
        x, y = 4, 3

    return x, y


def get_position_board_1v1(pos):

    x, y = None, None

    if pos == 0:
        x, y = 0, 0
    if pos == 1:
        x, y = 1, 0
    if pos == 2:
        x, y = 2, 0
    if pos == 3:
        x, y = 3, 0
    if pos == 4:
        x, y = 4, 0
    if pos == 5:
        x, y = 5, 0
    if pos == 6:
        x, y = 0, 1
    if pos == 7:
        x, y = 1, 1
    if pos == 8:
        x, y = 2, 1
    if pos == 9:
        x, y = 3, 1
    if pos == 10:
        x, y = 4, 1
    if pos == 11:
        x, y = 5, 1
    if pos == 12:
        x, y = 0, 2
    if pos == 13:
        x, y = 1, 2
    if pos == 14:
        x, y = 2, 2
    if pos == 15:
        x, y = 3, 2
    if pos == 16:
        x, y = 4, 2
    if pos == 17:
        x, y = 5, 2
    if pos == 18:
        x, y = 0, 3
    if pos == 19:
        x, y = 1, 3
    if pos == 20:
        x, y = 2, 3
    if pos == 21:
        x, y = 3, 3
    if pos == 22:
        x, y = 4, 3
    if pos == 23:
        x, y = 5, 3
    if pos == 24:
        x, y = 0, 4
    if pos == 25:
        x, y = 1, 4
    if pos == 26:
        x, y = 2, 4
    if pos == 27:
        x, y = 3, 4
    if pos == 28:
        x, y = 4, 4
    if pos == 29:
        x, y = 5, 4

    return x, y


def get_position_field_3v3(pos):

    x = int(pos % 12) if pos is not None else None
    y = int(pos / 12) if pos is not None else None

    x += 1
    y += 1

    return x, y


def get_position_board_3v3(pos):

    x = int(pos % 14) if pos is not None else None
    y = int(pos / 14) if pos is not None else None

    return x, y


def create_action_dictionary_1v1():

    actions_dictionary = []

    START_GAME = {
        'action_index': 0,
        'positions_start': 0,
        'positions_end': 0
    }

    actions_dictionary.append(START_GAME)

    HEADS = {
        'action_index': 1,
        'positions_start': 1,
        'positions_end': 1
    }
    actions_dictionary.append(HEADS)

    TAILS = {
        'action_index': 2,
        'positions_start': 2,
        'positions_end': 2
    }
    actions_dictionary.append(TAILS)

    KICK = {
        'action_index': 3,
        'positions_start': 3,
        'positions_end': 3
    }
    actions_dictionary.append(KICK)

    RECIEVE = {
        'action_index': 4,
        'positions_start': 4,
        'positions_end': 4
    }
    actions_dictionary.append(RECIEVE)

    END_PLAYER_TURN = {
        'action_index': 5,
        'positions_start': 5,
        'positions_end': 5
    }
    actions_dictionary.append(END_PLAYER_TURN)

    USE_REROLL = {
        'action_index': 6,
        'positions_start': 6,
        'positions_end': 6
    }
    actions_dictionary.append(USE_REROLL)

    DONT_USE_REROLL = {
        'action_index': 7,
        'positions_start': 7,
        'positions_end': 7
    }
    actions_dictionary.append(DONT_USE_REROLL)

    END_TURN = {
        'action_index': 8,
        'positions_start': 8,
        'positions_end': 8
    }
    actions_dictionary.append(END_TURN)

    SELECT_ATTACKER_DOWN = {
        'action_index': 10,
        'positions_start': 9,
        'positions_end': 9
    }
    actions_dictionary.append(SELECT_ATTACKER_DOWN)

    SELECT_BOTH_DOWN = {
        'action_index': 11,
        'positions_start': 10,
        'positions_end': 10
    }
    actions_dictionary.append(SELECT_BOTH_DOWN)

    SELECT_PUSH = {
        'action_index': 12,
        'positions_start': 11,
        'positions_end': 11
    }
    actions_dictionary.append(SELECT_PUSH)

    SELECT_DEFENDER_STUMBLES = {
        'action_index': 13,
        'positions_start': 12,
        'positions_end': 12
    }
    actions_dictionary.append(SELECT_DEFENDER_STUMBLES)

    SELECT_DEFENDER_DOWN = {
        'action_index': 14,
        'positions_start': 13,
        'positions_end': 13
    }
    actions_dictionary.append(SELECT_DEFENDER_DOWN)

    SELECT_NONE = {
        'action_index': 15,
        'positions_start': 14,
        'positions_end': 14
    }
    actions_dictionary.append(SELECT_NONE)

    INTERCEPTION = {
        'action_index': 20,
        'positions_start': 15,
        'positions_end': 15
    }
    actions_dictionary.append(INTERCEPTION)

    SETUP_FORM_WEDGE = {
        'action_index': 33,
        'positions_start': 16,
        'positions_end': 16
    }
    actions_dictionary.append(SETUP_FORM_WEDGE)

    SETUP_FORM_LINE = {
        'action_index': 34,
        'positions_start': 17,
        'positions_end': 17,

    }
    actions_dictionary.append(SETUP_FORM_LINE)

    SETUP_FORM_SPREAD = {
        'action_index': 35,
        'positions_start': 18,
        'positions_end': 18
    }
    actions_dictionary.append(SETUP_FORM_SPREAD)

    SETUP_FORM_ZONE = {
        'action_index': 36,
        'positions_start': 19,
        'positions_end': 19
    }
    actions_dictionary.append(SETUP_FORM_ZONE)

    # SPATIAL ACTIONS

    STAND_UP = {
        'action_index': 9,
        'positions_start': 20,
        'positions_end': 31
    }
    actions_dictionary.append(STAND_UP)

    PLACE_PLAYER = {
        'action_index': 16,
        'positions_start': 32,
        'positions_end': 43
    }
    actions_dictionary.append(PLACE_PLAYER)

    PLACE_BALL = {
        'action_index': 17,
        'positions_start': 44,
        'positions_end': 55
    }
    actions_dictionary.append(PLACE_BALL)

    PUSH = {
        'action_index': 18,
        'positions_start': 56,
        'positions_end': 85
    }
    actions_dictionary.append(PUSH)

    FOLLOW_UP = {
        'action_index': 19,
        'positions_start': 86,
        'positions_end': 97
    }
    actions_dictionary.append(FOLLOW_UP)

    SELECT_PLAYER = {
        'action_index': 21,
        'positions_start': 98,
        'positions_end': 109
    }
    actions_dictionary.append(SELECT_PLAYER)

    MOVE = {
        'action_index': 22,
        'positions_start': 110,
        'positions_end': 121
    }
    actions_dictionary.append(MOVE)

    BLOCK = {
        'action_index': 23,
        'positions_start': 122,
        'positions_end': 133
    }
    actions_dictionary.append(BLOCK)

    PASS = {
        'action_index': 24,
        'positions_start': 134,
        'positions_end': 145
    }
    actions_dictionary.append(PASS)

    FOUL = {
        'action_index': 25,
        'positions_start': 146,
        'positions_end': 157
    }
    actions_dictionary.append(FOUL)

    HANDOFF = {
        'action_index': 26,
        'positions_start': 158,
        'positions_end': 169
    }
    actions_dictionary.append(HANDOFF)

    START_MOVE = {
        'action_index': 27,
        'positions_start': 170,
        'positions_end': 181
    }
    actions_dictionary.append(START_MOVE)

    START_BLOCK = {
        'action_index': 28,
        'positions_start': 182,
        'positions_end': 193
    }
    actions_dictionary.append(START_BLOCK)

    START_BLITZ = {
        'action_index': 29,
        'positions_start': 194,
        'positions_end': 205
    }
    actions_dictionary.append(START_BLITZ)

    START_PASS = {
        'action_index': 30,
        'positions_start': 206,
        'positions_end': 217
    }
    actions_dictionary.append(START_PASS)

    START_FOUL = {
        'action_index': 31,
        'positions_start': 218,
        'positions_end': 229
    }
    actions_dictionary.append(START_FOUL)

    START_HANDOFF = {
        'action_index': 32,
        'positions_start': 230,
        'positions_end': 241
    }
    actions_dictionary.append(START_HANDOFF)


    return actions_dictionary


    # ALL_ACTIONS = {
    #     0: START_GAME,
    #     1: HEADS,
    #     2: TAILS,
    #     3: KICK,
    #     4: RECIEVE,
    #     5: END_PLAYER_TURN,
    #     6: USE_REROLL,
    #     7: DONT_USE_REROLL,
    #     8: END_TURN,
    #     10: SELECT_ATTACKER_DOWN,
    #     11: SELECT_BOTH_DOWN,
    #     12: SELECT_PUSH,
    #     13: SELECT_DEFENDER_STUMBLES,
    #     14: SELECT_DEFENDER_DOWN,
    #     15: SELECT_NONE,
    #     16: PLACE_PLAYER,
    #     17: PLACE_BALL,
    #     18: PUSH,
    #     19: FOLLOW_UP,
    #     20: INTERCEPTION,
    #     21: SELECT_PLAYER,
    #     22: MOVE,
    #     23: BLOCK,
    #     24: PASS,
    #     25: FOUL,
    #     26: HANDOFF,
    #     27: START_MOVE,
    #     28: START_BLOCK,
    #     29: START_BLITZ,
    #     30: START_PASS,
    #     31: START_FOUL,
    #     32: START_HANDOFF,
    #     33: SETUP_FORM_WEDGE,
    #     34: SETUP_FORM_LINE,
    #     35: SETUP_FORM_SPREAD,
    #     36: SETUP_FORM_ZONE
    # }


def create_action_dictionary_1v1_full_board():

    actions_dictionary = []

    START_GAME = {
        'action_index': 0,
        'positions_start': 0,
        'positions_end': 0
    }

    actions_dictionary.append(START_GAME)

    HEADS = {
        'action_index': 1,
        'positions_start': 1,
        'positions_end': 1
    }
    actions_dictionary.append(HEADS)

    TAILS = {
        'action_index': 2,
        'positions_start': 2,
        'positions_end': 2
    }
    actions_dictionary.append(TAILS)

    KICK = {
        'action_index': 3,
        'positions_start': 3,
        'positions_end': 3
    }
    actions_dictionary.append(KICK)

    RECIEVE = {
        'action_index': 4,
        'positions_start': 4,
        'positions_end': 4
    }
    actions_dictionary.append(RECIEVE)

    END_PLAYER_TURN = {
        'action_index': 5,
        'positions_start': 5,
        'positions_end': 5
    }
    actions_dictionary.append(END_PLAYER_TURN)

    USE_REROLL = {
        'action_index': 6,
        'positions_start': 6,
        'positions_end': 6
    }
    actions_dictionary.append(USE_REROLL)

    DONT_USE_REROLL = {
        'action_index': 7,
        'positions_start': 7,
        'positions_end': 7
    }
    actions_dictionary.append(DONT_USE_REROLL)

    END_TURN = {
        'action_index': 8,
        'positions_start': 8,
        'positions_end': 8
    }
    actions_dictionary.append(END_TURN)

    SELECT_ATTACKER_DOWN = {
        'action_index': 10,
        'positions_start': 9,
        'positions_end': 9
    }
    actions_dictionary.append(SELECT_ATTACKER_DOWN)

    SELECT_BOTH_DOWN = {
        'action_index': 11,
        'positions_start': 10,
        'positions_end': 10
    }
    actions_dictionary.append(SELECT_BOTH_DOWN)

    SELECT_PUSH = {
        'action_index': 12,
        'positions_start': 11,
        'positions_end': 11
    }
    actions_dictionary.append(SELECT_PUSH)

    SELECT_DEFENDER_STUMBLES = {
        'action_index': 13,
        'positions_start': 12,
        'positions_end': 12
    }
    actions_dictionary.append(SELECT_DEFENDER_STUMBLES)

    SELECT_DEFENDER_DOWN = {
        'action_index': 14,
        'positions_start': 13,
        'positions_end': 13
    }
    actions_dictionary.append(SELECT_DEFENDER_DOWN)

    SELECT_NONE = {
        'action_index': 15,
        'positions_start': 14,
        'positions_end': 14
    }
    actions_dictionary.append(SELECT_NONE)

    INTERCEPTION = {
        'action_index': 20,
        'positions_start': 15,
        'positions_end': 15
    }
    actions_dictionary.append(INTERCEPTION)

    SETUP_FORM_WEDGE = {
        'action_index': 33,
        'positions_start': 16,
        'positions_end': 16
    }
    actions_dictionary.append(SETUP_FORM_WEDGE)

    SETUP_FORM_LINE = {
        'action_index': 34,
        'positions_start': 17,
        'positions_end': 17,

    }
    actions_dictionary.append(SETUP_FORM_LINE)

    SETUP_FORM_SPREAD = {
        'action_index': 35,
        'positions_start': 18,
        'positions_end': 18
    }
    actions_dictionary.append(SETUP_FORM_SPREAD)

    SETUP_FORM_ZONE = {
        'action_index': 36,
        'positions_start': 19,
        'positions_end': 19
    }
    actions_dictionary.append(SETUP_FORM_ZONE)

    # SPATIAL ACTIONS

    STAND_UP = {
        'action_index': 9,
        'positions_start': 20,
    }
    actions_dictionary.append(STAND_UP)

    PLACE_PLAYER = {
        'action_index': 16,
        'positions_start': 50,
    }
    actions_dictionary.append(PLACE_PLAYER)

    PLACE_BALL = {
        'action_index': 17,
        'positions_start': 80,
    }
    actions_dictionary.append(PLACE_BALL)

    PUSH = {
        'action_index': 18,
        'positions_start': 110,
    }
    actions_dictionary.append(PUSH)

    FOLLOW_UP = {
        'action_index': 19,
        'positions_start': 140,
    }
    actions_dictionary.append(FOLLOW_UP)

    SELECT_PLAYER = {
        'action_index': 21,
        'positions_start': 170,
    }
    actions_dictionary.append(SELECT_PLAYER)

    MOVE = {
        'action_index': 22,
        'positions_start': 200,
    }
    actions_dictionary.append(MOVE)

    BLOCK = {
        'action_index': 23,
        'positions_start': 230,

    }
    actions_dictionary.append(BLOCK)

    PASS = {
        'action_index': 24,
        'positions_start': 260,
    }
    actions_dictionary.append(PASS)

    FOUL = {
        'action_index': 25,
        'positions_start': 290,
    }
    actions_dictionary.append(FOUL)

    HANDOFF = {
        'action_index': 26,
        'positions_start': 320,
    }
    actions_dictionary.append(HANDOFF)

    START_MOVE = {
        'action_index': 27,
        'positions_start': 350,
    }
    actions_dictionary.append(START_MOVE)

    START_BLOCK = {
        'action_index': 28,
        'positions_start': 380,
    }
    actions_dictionary.append(START_BLOCK)

    START_BLITZ = {
        'action_index': 29,
        'positions_start': 410,
    }
    actions_dictionary.append(START_BLITZ)

    START_PASS = {
        'action_index': 30,
        'positions_start': 440,
    }
    actions_dictionary.append(START_PASS)

    START_FOUL = {
        'action_index': 31,
        'positions_start': 470,
    }
    actions_dictionary.append(START_FOUL)

    START_HANDOFF = {
        'action_index': 32,
        'positions_start': 500,
    }
    actions_dictionary.append(START_HANDOFF)

    return actions_dictionary


    # ALL_ACTIONS = {
    #     0: START_GAME,
    #     1: HEADS,
    #     2: TAILS,
    #     3: KICK,
    #     4: RECIEVE,
    #     5: END_PLAYER_TURN,
    #     6: USE_REROLL,
    #     7: DONT_USE_REROLL,
    #     8: END_TURN,
    #     10: SELECT_ATTACKER_DOWN,
    #     11: SELECT_BOTH_DOWN,
    #     12: SELECT_PUSH,
    #     13: SELECT_DEFENDER_STUMBLES,
    #     14: SELECT_DEFENDER_DOWN,
    #     15: SELECT_NONE,
    #     16: PLACE_PLAYER,
    #     17: PLACE_BALL,
    #     18: PUSH,
    #     19: FOLLOW_UP,
    #     20: INTERCEPTION,
    #     21: SELECT_PLAYER,
    #     22: MOVE,
    #     23: BLOCK,
    #     24: PASS,
    #     25: FOUL,
    #     26: HANDOFF,
    #     27: START_MOVE,
    #     28: START_BLOCK,
    #     29: START_BLITZ,
    #     30: START_PASS,
    #     31: START_FOUL,
    #     32: START_HANDOFF,
    #     33: SETUP_FORM_WEDGE,
    #     34: SETUP_FORM_LINE,
    #     35: SETUP_FORM_SPREAD,
    #     36: SETUP_FORM_ZONE
    # }


def create_action_dictionary_3v3():

    actions_dictionary = []

    START_GAME = {
        'action_index': 0,
        'positions_start': 0,
        'positions_end': 0
    }

    actions_dictionary.append(START_GAME)

    HEADS = {
        'action_index': 1,
        'positions_start': 1,
        'positions_end': 1
    }
    actions_dictionary.append(HEADS)

    TAILS = {
        'action_index': 2,
        'positions_start': 2,
        'positions_end': 2
    }
    actions_dictionary.append(TAILS)

    KICK = {
        'action_index': 3,
        'positions_start': 3,
        'positions_end': 3
    }
    actions_dictionary.append(KICK)

    RECIEVE = {
        'action_index': 4,
        'positions_start': 4,
        'positions_end': 4
    }
    actions_dictionary.append(RECIEVE)

    END_PLAYER_TURN = {
        'action_index': 5,
        'positions_start': 5,
        'positions_end': 5
    }
    actions_dictionary.append(END_PLAYER_TURN)

    USE_REROLL = {
        'action_index': 6,
        'positions_start': 6,
        'positions_end': 6
    }
    actions_dictionary.append(USE_REROLL)

    DONT_USE_REROLL = {
        'action_index': 7,
        'positions_start': 7,
        'positions_end': 7
    }
    actions_dictionary.append(DONT_USE_REROLL)

    END_TURN = {
        'action_index': 8,
        'positions_start': 8,
        'positions_end': 8
    }
    actions_dictionary.append(END_TURN)

    SELECT_ATTACKER_DOWN = {
        'action_index': 10,
        'positions_start': 9,
        'positions_end': 9
    }
    actions_dictionary.append(SELECT_ATTACKER_DOWN)

    SELECT_BOTH_DOWN = {
        'action_index': 11,
        'positions_start': 10,
        'positions_end': 10
    }
    actions_dictionary.append(SELECT_BOTH_DOWN)

    SELECT_PUSH = {
        'action_index': 12,
        'positions_start': 11,
        'positions_end': 11
    }
    actions_dictionary.append(SELECT_PUSH)

    SELECT_DEFENDER_STUMBLES = {
        'action_index': 13,
        'positions_start': 12,
        'positions_end': 12
    }
    actions_dictionary.append(SELECT_DEFENDER_STUMBLES)

    SELECT_DEFENDER_DOWN = {
        'action_index': 14,
        'positions_start': 13,
        'positions_end': 13
    }
    actions_dictionary.append(SELECT_DEFENDER_DOWN)

    SELECT_NONE = {
        'action_index': 15,
        'positions_start': 14,
        'positions_end': 14
    }
    actions_dictionary.append(SELECT_NONE)

    INTERCEPTION = {
        'action_index': 20,
        'positions_start': 15,
        'positions_end': 15
    }
    actions_dictionary.append(INTERCEPTION)

    SETUP_FORM_WEDGE = {
        'action_index': 33,
        'positions_start': 16,
        'positions_end': 16
    }
    actions_dictionary.append(SETUP_FORM_WEDGE)

    SETUP_FORM_LINE = {
        'action_index': 34,
        'positions_start': 17,
        'positions_end': 17,

    }
    actions_dictionary.append(SETUP_FORM_LINE)

    SETUP_FORM_SPREAD = {
        'action_index': 35,
        'positions_start': 18,
        'positions_end': 18
    }
    actions_dictionary.append(SETUP_FORM_SPREAD)

    SETUP_FORM_ZONE = {
        'action_index': 36,
        'positions_start': 19,
        'positions_end': 19
    }
    actions_dictionary.append(SETUP_FORM_ZONE)

    # SPATIAL ACTIONS

    STAND_UP = {
        'action_index': 9,
        'positions_start': 20,
        'positions_end': 79
    }
    actions_dictionary.append(STAND_UP)

    PLACE_PLAYER = {
        'action_index': 16,
        'positions_start': 80,
        'positions_end': 139
    }
    actions_dictionary.append(PLACE_PLAYER)

    PLACE_BALL = {
        'action_index': 17,
        'positions_start': 140,
        'positions_end': 199
    }
    actions_dictionary.append(PLACE_BALL)

    PUSH = {
        'action_index': 18,
        'positions_start': 200,
        'positions_end': 297
    }
    actions_dictionary.append(PUSH)

    FOLLOW_UP = {
        'action_index': 19,
        'positions_start': 298,
        'positions_end': 357
    }
    actions_dictionary.append(FOLLOW_UP)

    SELECT_PLAYER = {
        'action_index': 21,
        'positions_start': 358,
        'positions_end': 417
    }
    actions_dictionary.append(SELECT_PLAYER)

    MOVE = {
        'action_index': 22,
        'positions_start': 418,
        'positions_end': 477
    }
    actions_dictionary.append(MOVE)

    BLOCK = {
        'action_index': 23,
        'positions_start': 478,
        'positions_end': 537
    }
    actions_dictionary.append(BLOCK)

    PASS = {
        'action_index': 24,
        'positions_start': 538,
        'positions_end': 597
    }
    actions_dictionary.append(PASS)

    FOUL = {
        'action_index': 25,
        'positions_start': 598,
        'positions_end': 657
    }
    actions_dictionary.append(FOUL)

    HANDOFF = {
        'action_index': 26,
        'positions_start': 658,
        'positions_end': 717
    }
    actions_dictionary.append(HANDOFF)

    START_MOVE = {
        'action_index': 27,
        'positions_start': 718,
        'positions_end': 777
    }
    actions_dictionary.append(START_MOVE)

    START_BLOCK = {
        'action_index': 28,
        'positions_start': 778,
        'positions_end': 837
    }
    actions_dictionary.append(START_BLOCK)

    START_BLITZ = {
        'action_index': 29,
        'positions_start': 838,
        'positions_end': 897
    }
    actions_dictionary.append(START_BLITZ)

    START_PASS = {
        'action_index': 30,
        'positions_start': 898,
        'positions_end': 957
    }
    actions_dictionary.append(START_PASS)

    START_FOUL = {
        'action_index': 31,
        'positions_start': 958,
        'positions_end': 1017
    }
    actions_dictionary.append(START_FOUL)

    START_HANDOFF = {
        'action_index': 32,
        'positions_start': 1018,
        'positions_end': 1077
    }
    actions_dictionary.append(START_HANDOFF)


    return actions_dictionary


    # ALL_ACTIONS = {
    #     0: START_GAME,
    #     1: HEADS,
    #     2: TAILS,
    #     3: KICK,
    #     4: RECIEVE,
    #     5: END_PLAYER_TURN,
    #     6: USE_REROLL,
    #     7: DONT_USE_REROLL,
    #     8: END_TURN,
    #     10: SELECT_ATTACKER_DOWN,
    #     11: SELECT_BOTH_DOWN,
    #     12: SELECT_PUSH,
    #     13: SELECT_DEFENDER_STUMBLES,
    #     14: SELECT_DEFENDER_DOWN,
    #     15: SELECT_NONE,
    #     16: PLACE_PLAYER,
    #     17: PLACE_BALL,
    #     18: PUSH,
    #     19: FOLLOW_UP,
    #     20: INTERCEPTION,
    #     21: SELECT_PLAYER,
    #     22: MOVE,
    #     23: BLOCK,
    #     24: PASS,
    #     25: FOUL,
    #     26: HANDOFF,
    #     27: START_MOVE,
    #     28: START_BLOCK,
    #     29: START_BLITZ,
    #     30: START_PASS,
    #     31: START_FOUL,
    #     32: START_HANDOFF,
    #     33: SETUP_FORM_WEDGE,
    #     34: SETUP_FORM_LINE,
    #     35: SETUP_FORM_SPREAD,
    #     36: SETUP_FORM_ZONE
    # }


def create_action_dictionary_3v3_new_approach():

    actions_dictionary = []

    START_GAME = {
        'action_index': 0,
        'positions_start': 0,
        'positions_end': 0
    }

    actions_dictionary.append(START_GAME)

    HEADS = {
        'action_index': 1,
        'positions_start': 1,
        'positions_end': 1
    }
    actions_dictionary.append(HEADS)

    TAILS = {
        'action_index': 2,
        'positions_start': 2,
        'positions_end': 2
    }
    actions_dictionary.append(TAILS)

    KICK = {
        'action_index': 3,
        'positions_start': 3,
        'positions_end': 3
    }
    actions_dictionary.append(KICK)

    RECIEVE = {
        'action_index': 4,
        'positions_start': 4,
        'positions_end': 4
    }
    actions_dictionary.append(RECIEVE)

    END_PLAYER_TURN = {
        'action_index': 5,
        'positions_start': 5,
        'positions_end': 5
    }
    actions_dictionary.append(END_PLAYER_TURN)

    USE_REROLL = {
        'action_index': 6,
        'positions_start': 6,
        'positions_end': 6
    }
    actions_dictionary.append(USE_REROLL)

    DONT_USE_REROLL = {
        'action_index': 7,
        'positions_start': 7,
        'positions_end': 7
    }
    actions_dictionary.append(DONT_USE_REROLL)

    END_TURN = {
        'action_index': 8,
        'positions_start': 8,
        'positions_end': 8
    }
    actions_dictionary.append(END_TURN)

    SELECT_ATTACKER_DOWN = {
        'action_index': 10,
        'positions_start': 9,
        'positions_end': 9
    }
    actions_dictionary.append(SELECT_ATTACKER_DOWN)

    SELECT_BOTH_DOWN = {
        'action_index': 11,
        'positions_start': 10,
        'positions_end': 10
    }
    actions_dictionary.append(SELECT_BOTH_DOWN)

    SELECT_PUSH = {
        'action_index': 12,
        'positions_start': 11,
        'positions_end': 11
    }
    actions_dictionary.append(SELECT_PUSH)

    SELECT_DEFENDER_STUMBLES = {
        'action_index': 13,
        'positions_start': 12,
        'positions_end': 12
    }
    actions_dictionary.append(SELECT_DEFENDER_STUMBLES)

    SELECT_DEFENDER_DOWN = {
        'action_index': 14,
        'positions_start': 13,
        'positions_end': 13
    }
    actions_dictionary.append(SELECT_DEFENDER_DOWN)

    SELECT_NONE = {
        'action_index': 15,
        'positions_start': 14,
        'positions_end': 14
    }
    actions_dictionary.append(SELECT_NONE)

    INTERCEPTION = {
        'action_index': 20,
        'positions_start': 15,
        'positions_end': 15
    }
    actions_dictionary.append(INTERCEPTION)

    SETUP_FORM_WEDGE = {
        'action_index': 33,
        'positions_start': 16,
        'positions_end': 16
    }
    actions_dictionary.append(SETUP_FORM_WEDGE)

    SETUP_FORM_LINE = {
        'action_index': 34,
        'positions_start': 17,
        'positions_end': 17,

    }
    actions_dictionary.append(SETUP_FORM_LINE)

    SETUP_FORM_SPREAD = {
        'action_index': 35,
        'positions_start': 18,
        'positions_end': 18
    }
    actions_dictionary.append(SETUP_FORM_SPREAD)

    SETUP_FORM_ZONE = {
        'action_index': 36,
        'positions_start': 19,
        'positions_end': 19
    }
    actions_dictionary.append(SETUP_FORM_ZONE)

    # SPATIAL ACTIONS

    STAND_UP = {
        'action_index': 9,
        'positions_start': 20,
        'position_type': 'players'
    }
    actions_dictionary.append(STAND_UP)

    PLACE_PLAYER = {
        'action_index': 16,
        'positions_start': 25,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PLACE_PLAYER)

    PLACE_BALL = {
        'action_index': 17,
        'positions_start': 123,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PLACE_BALL)

    PUSH = {
        'action_index': 18,
        'positions_start': 221,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PUSH)

    FOLLOW_UP = {
        'action_index': 19,
        'positions_start': 319,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(FOLLOW_UP)

    SELECT_PLAYER = {
        'action_index': 21,
        'positions_start': 327,
        'position_type': 'players'
    }
    actions_dictionary.append(SELECT_PLAYER)

    MOVE = {
        'action_index': 22,
        'positions_start': 332,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(MOVE)

    BLOCK = {
        'action_index': 23,
        'positions_start': 340,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(BLOCK)

    PASS = {
        'action_index': 24,
        'positions_start': 348,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PASS)

    FOUL = {
        'action_index': 25,
        'positions_start': 446,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(FOUL)

    HANDOFF = {
        'action_index': 26,
        'positions_start': 454,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(HANDOFF)

    START_MOVE = {
        'action_index': 27,
        'positions_start': 462,
        'position_type': 'players'
    }
    actions_dictionary.append(START_MOVE)

    START_BLOCK = {
        'action_index': 28,
        'positions_start': 467,
        'position_type': 'players'
    }
    actions_dictionary.append(START_BLOCK)

    START_BLITZ = {
        'action_index': 29,
        'positions_start': 472,
        'position_type': 'players'
    }
    actions_dictionary.append(START_BLITZ)

    START_PASS = {
        'action_index': 30,
        'positions_start': 477,
        'position_type': 'players'
    }
    actions_dictionary.append(START_PASS)

    START_FOUL = {
        'action_index': 31,
        'positions_start': 482,
        'position_type': 'players'
    }
    actions_dictionary.append(START_FOUL)

    START_HANDOFF = {
        'action_index': 32,
        'positions_start': 487,
        'position_type': 'players'
    }
    actions_dictionary.append(START_HANDOFF)

    return actions_dictionary


def map_actions_3v3_new_approach(actions, players, own_players):
    mapped_actions = []
    x_positions = []
    y_positions = []

    for act, player, own_player in zip(actions, players, own_players):
        action = act.item()
        mapped_action = 0
        x, y = None, None

        # NON-SPATIAL ACTIONS
        if action == 0:
            mapped_action = 0
        if action == 1:
            mapped_action = 1
        if action == 2:
            mapped_action = 2
        if action == 3:
            mapped_action = 3
        if action == 4:
            mapped_action = 4
        if action == 5:
            mapped_action = 5
        if action == 6:
            mapped_action = 6
        if action == 7:
            mapped_action = 7
        if action == 8:
            mapped_action = 8
        if action == 9:
            mapped_action = 10
        if action == 10:
            mapped_action = 11
        if action == 11:
            mapped_action = 12
        if action == 12:
            mapped_action = 13
        if action == 13:
            mapped_action = 14
        if action == 14:
            mapped_action = 15
        if action == 15:
            mapped_action = 20
        if action == 16:
            mapped_action = 33
        if action == 17:
            mapped_action = 34
        if action == 18:
            mapped_action = 35
        if action == 19:
            mapped_action = 36

        # SPATIAL ACTIONS
        if 20 <= action <= (20+4):  # STAND-UP
            mapped_action = 9
            index = action - 20
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)

        if 25 <= action <= (25+97):  # PLACE PLAYER
            mapped_action = 16
            position = action - 25
            x, y = get_position_board_3v3(position)
        if 123 <= action <= (123+97):  # PLACE BALL
            mapped_action = 17
            position = action - 123
            x, y = get_position_board_3v3(position)

        if 221 <= action <= (221+97):  # PUSH
            mapped_action = 18
            position = action - 221
            x, y = get_position_board_3v3(position)

        if 319 <= action <= (319+7):  # FOLLOW-UP
            mapped_action = 19
            index = action - 319
            x, y = get_positions_active(index, player)

        if 327 <= action <= (327 + 4):  # SELECT PLAYER
            mapped_action = 21
            index = action - 327
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)

        if 332 <= action <= (332+7):  # MOVE
            mapped_action = 22
            index = action - 332
            x, y = get_positions_active(index, player)

        if 340 <= action <= (340+7):  # BLOCK
            mapped_action = 23
            index = action - 340
            x, y = get_positions_active(index, player)

        if 348 <= action <= (348+97):  # PASS
            mapped_action = 24
            position = action - 348
            x, y = get_position_board_3v3(position)

        if 446 <= action <= (446+7):  # FOUL
            mapped_action = 25
            index = action - 446
            x, y = get_positions_active(index, player)
        if 454 <= action <= (454+7):  # HANDOFF
            mapped_action = 26
            index = action - 454
            x, y = get_positions_active(index, player)

        if 462 <= action <= (462+4):  # START MOVE
            mapped_action = 27
            index = action - 462
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)
        if 467 <= action <= (467+4):  # START BLOCK
            mapped_action = 28
            index = action - 467
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)
        if 472 <= action <= (472+4):  # START BLITZ
            mapped_action = 29
            index = action - 472
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)
        if 477 <= action <= (477+4):  # START PASS
            mapped_action = 30
            index = action - 477
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)
        if 482 <= action <= (482+4):  # START FOUL
            mapped_action = 31
            index = action - 482
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)
        if 487 <= action <= (487+4):  # START HANDOFF
            mapped_action = 32
            index = action - 487
            o_player = own_player[index]
            x = int(o_player % 14)
            y = int(o_player / 14)

        mapped_actions.append(mapped_action)
        x_positions.append(x)
        y_positions.append(y)

    return mapped_actions, x_positions, y_positions


def get_positions_active(index, player):
    if index == 0:
        new_index = player - 15
    elif index == 1:
        new_index = player - 14
    elif index == 2:
        new_index = player - 13
    elif index == 3:
        new_index = player - 1
    elif index == 4:
        new_index = player + 1
    elif index == 5:
        new_index = player + 13
    elif index == 6:
        new_index = player + 14
    elif index == 7:
        new_index = player + 15

    try:
        x = int(new_index % 14)
    except:
        print()
    y = int(new_index / 14)

    return x, y

def get_position_board_5v5(pos):

    x = int(pos % 18) if pos is not None else None
    y = int(pos / 18) if pos is not None else None

    return x, y

def get_positions_active_5v5(index, player):
    if index == 0:
        new_index = player - 19
    elif index == 1:
        new_index = player - 18
    elif index == 2:
        new_index = player - 17
    elif index == 3:
        new_index = player - 1
    elif index == 4:
        new_index = player + 1
    elif index == 5:
        new_index = player + 17
    elif index == 6:
        new_index = player + 18
    elif index == 7:
        new_index = player + 19

    x = int(new_index % 18)
    y = int(new_index / 18)

    return x, y

def create_action_dictionary_5v5_pruned():

    actions_dictionary = []

    START_GAME = {
        'action_index': 0,
        'positions_start': 0,
        'positions_end': 0
    }

    actions_dictionary.append(START_GAME)

    HEADS = {
        'action_index': 1,
        'positions_start': 1,
        'positions_end': 1
    }
    actions_dictionary.append(HEADS)

    TAILS = {
        'action_index': 2,
        'positions_start': 2,
        'positions_end': 2
    }
    actions_dictionary.append(TAILS)

    KICK = {
        'action_index': 3,
        'positions_start': 3,
        'positions_end': 3
    }
    actions_dictionary.append(KICK)

    RECIEVE = {
        'action_index': 4,
        'positions_start': 4,
        'positions_end': 4
    }
    actions_dictionary.append(RECIEVE)

    END_PLAYER_TURN = {
        'action_index': 5,
        'positions_start': 5,
        'positions_end': 5
    }
    actions_dictionary.append(END_PLAYER_TURN)

    USE_REROLL = {
        'action_index': 6,
        'positions_start': 6,
        'positions_end': 6
    }
    actions_dictionary.append(USE_REROLL)

    DONT_USE_REROLL = {
        'action_index': 7,
        'positions_start': 7,
        'positions_end': 7
    }
    actions_dictionary.append(DONT_USE_REROLL)

    END_TURN = {
        'action_index': 8,
        'positions_start': 8,
        'positions_end': 8
    }
    actions_dictionary.append(END_TURN)

    SELECT_ATTACKER_DOWN = {
        'action_index': 10,
        'positions_start': 9,
        'positions_end': 9
    }
    actions_dictionary.append(SELECT_ATTACKER_DOWN)

    SELECT_BOTH_DOWN = {
        'action_index': 11,
        'positions_start': 10,
        'positions_end': 10
    }
    actions_dictionary.append(SELECT_BOTH_DOWN)

    SELECT_PUSH = {
        'action_index': 12,
        'positions_start': 11,
        'positions_end': 11
    }
    actions_dictionary.append(SELECT_PUSH)

    SELECT_DEFENDER_STUMBLES = {
        'action_index': 13,
        'positions_start': 12,
        'positions_end': 12
    }
    actions_dictionary.append(SELECT_DEFENDER_STUMBLES)

    SELECT_DEFENDER_DOWN = {
        'action_index': 14,
        'positions_start': 13,
        'positions_end': 13
    }
    actions_dictionary.append(SELECT_DEFENDER_DOWN)

    SELECT_NONE = {
        'action_index': 15,
        'positions_start': 14,
        'positions_end': 14
    }
    actions_dictionary.append(SELECT_NONE)

    INTERCEPTION = {
        'action_index': 20,
        'positions_start': 15,
        'positions_end': 15
    }
    actions_dictionary.append(INTERCEPTION)

    SETUP_FORM_WEDGE = {
        'action_index': 33,
        'positions_start': 16,
        'positions_end': 16
    }
    actions_dictionary.append(SETUP_FORM_WEDGE)

    SETUP_FORM_LINE = {
        'action_index': 34,
        'positions_start': 17,
        'positions_end': 17,

    }
    actions_dictionary.append(SETUP_FORM_LINE)

    SETUP_FORM_SPREAD = {
        'action_index': 35,
        'positions_start': 18,
        'positions_end': 18
    }
    actions_dictionary.append(SETUP_FORM_SPREAD)

    SETUP_FORM_ZONE = {
        'action_index': 36,
        'positions_start': 19,
        'positions_end': 19
    }
    actions_dictionary.append(SETUP_FORM_ZONE)

    # SPATIAL ACTIONS

    STAND_UP = {
        'action_index': 9,
        'positions_start': 20,
        'position_type': 'players'
    }
    actions_dictionary.append(STAND_UP)

    PLACE_PLAYER = {
        'action_index': 16,
        'positions_start': 27,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PLACE_PLAYER)

    PLACE_BALL = {
        'action_index': 17,
        'positions_start': 225,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PLACE_BALL)

    PUSH = {
        'action_index': 18,
        'positions_start': 423,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PUSH)

    FOLLOW_UP = {
        'action_index': 19,
        'positions_start': 621,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(FOLLOW_UP)

    SELECT_PLAYER = {
        'action_index': 21,
        'positions_start': 629,
        'position_type': 'players'
    }
    actions_dictionary.append(SELECT_PLAYER)

    MOVE = {
        'action_index': 22,
        'positions_start': 636,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(MOVE)

    BLOCK = {
        'action_index': 23,
        'positions_start': 644,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(BLOCK)

    PASS = {
        'action_index': 24,
        'positions_start': 652,
        'position_type': 'full_board'
    }
    actions_dictionary.append(PASS)

    FOUL = {
        'action_index': 25,
        'positions_start': 850,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(FOUL)

    HANDOFF = {
        'action_index': 26,
        'positions_start': 858,
        'position_type': 'adjacent'
    }
    actions_dictionary.append(HANDOFF)

    START_MOVE = {
        'action_index': 27,
        'positions_start': 866,
        'position_type': 'players'
    }
    actions_dictionary.append(START_MOVE)

    START_BLOCK = {
        'action_index': 28,
        'positions_start': 873,
        'position_type': 'players'
    }
    actions_dictionary.append(START_BLOCK)

    START_BLITZ = {
        'action_index': 29,
        'positions_start': 880,
        'position_type': 'players'
    }
    actions_dictionary.append(START_BLITZ)

    START_PASS = {
        'action_index': 30,
        'positions_start': 887,
        'position_type': 'players'
    }
    actions_dictionary.append(START_PASS)

    START_FOUL = {
        'action_index': 31,
        'positions_start': 894,
        'position_type': 'players'
    }
    actions_dictionary.append(START_FOUL)

    START_HANDOFF = {
        'action_index': 32,
        'positions_start': 901,
        'position_type': 'players'
    }
    actions_dictionary.append(START_HANDOFF)

    return actions_dictionary

def map_actions_5v5_pruned(actions, players, own_players):
    mapped_actions = []
    x_positions = []
    y_positions = []

    for act, player, own_player in zip(actions, players, own_players):
        action = act.item()
        mapped_action = 0
        x, y = None, None

        # NON-SPATIAL ACTIONS
        if action == 0:
            mapped_action = 0
        if action == 1:
            mapped_action = 1
        if action == 2:
            mapped_action = 2
        if action == 3:
            mapped_action = 3
        if action == 4:
            mapped_action = 4
        if action == 5:
            mapped_action = 5
        if action == 6:
            mapped_action = 6
        if action == 7:
            mapped_action = 7
        if action == 8:
            mapped_action = 8
        if action == 9:
            mapped_action = 10
        if action == 10:
            mapped_action = 11
        if action == 11:
            mapped_action = 12
        if action == 12:
            mapped_action = 13
        if action == 13:
            mapped_action = 14
        if action == 14:
            mapped_action = 15
        if action == 15:
            mapped_action = 20
        if action == 16:
            mapped_action = 33
        if action == 17:
            mapped_action = 34
        if action == 18:
            mapped_action = 35
        if action == 19:
            mapped_action = 36

        # SPATIAL ACTIONS
        if 20 <= action <= (20+6):  # STAND-UP
            mapped_action = 9
            index = action - 20
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)

        if 27 <= action <= (27+197):  # PLACE PLAYER
            mapped_action = 16
            position = action - 27
            x, y = get_position_board_5v5(position)
        if 225 <= action <= (225+197):  # PLACE BALL
            mapped_action = 17
            position = action - 225
            x, y = get_position_board_5v5(position)

        if 423 <= action <= (423+197):  # PUSH
            mapped_action = 18
            position = action - 423
            x, y = get_position_board_5v5(position)

        if 621 <= action <= (621+7):  # FOLLOW-UP
            mapped_action = 19
            index = action - 621
            x, y = get_positions_active_5v5(index, player)

        if 629 <= action <= (629 + 6):  # SELECT PLAYER
            mapped_action = 21
            index = action - 629
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)

        if 636 <= action <= (636+7):  # MOVE
            mapped_action = 22
            index = action - 636
            x, y = get_positions_active_5v5(index, player)

        if 644 <= action <= (644+7):  # BLOCK
            mapped_action = 23
            index = action - 644
            x, y = get_positions_active_5v5(index, player)

        if 652 <= action <= (652+197):  # PASS
            mapped_action = 24
            position = action - 652
            x, y = get_position_board_5v5(position)

        if 850 <= action <= (850+7):  # FOUL
            mapped_action = 25
            index = action - 850
            x, y = get_positions_active_5v5(index, player)
        if 858 <= action <= (858+7):  # HANDOFF
            mapped_action = 26
            index = action - 858
            x, y = get_positions_active_5v5(index, player)

        if 866 <= action <= (866+6):  # START MOVE
            mapped_action = 27
            index = action - 866
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)
        if 873 <= action <= (873+6):  # START BLOCK
            mapped_action = 28
            index = action - 873
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)
        if 880 <= action <= (880+6):  # START BLITZ
            mapped_action = 29
            index = action - 880
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)
        if 887 <= action <= (887+6):  # START PASS
            mapped_action = 30
            index = action - 887
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)
        if 894 <= action <= (894+6):  # START FOUL
            mapped_action = 31
            index = action - 894
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)
        if 901 <= action <= (901+6):  # START HANDOFF
            mapped_action = 32
            index = action - 901
            o_player = own_player[index]
            x = int(o_player % 18)
            y = int(o_player / 18)

        mapped_actions.append(mapped_action)
        x_positions.append(x)
        y_positions.append(y)

    return mapped_actions, x_positions, y_positions
