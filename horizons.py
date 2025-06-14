# Open field, no shading whatsoever
HORIZON_OPEN = [
	(0, 0)
]

# A balcony facing exactly south with a ~181° field of view from 90° to 270° (inclusive).
HORIZON_BALCONY_SOUTH = [
    (  0,  90),
	( 89,  90), # Values in between are linearly interpolated
	( 90,   0),
	(270,   0),
	(271,  90),
	(359,  90),
]

# A balcony facing exactly west with a ~181° field of view from 180° to 360°/0° (inclusive).
HORIZON_BALCONY_WEST = [
    (  0,   0),
	(  1,  90),
	(179,  90),
	(180,   0),
	(359,   0),
]

# A balcony facing exactly east with a ~181° field of view from 0° to 180° (inclusive).
HORIZON_BALCONY_EAST = [
    (  0,   0),
	(180,   0),
	(181,  90),
	(359,  90),
]

# A balcony facing exactly north with a ~181° field of view from 270° to 90° (inclusive).
HORIZON_BALCONY_NORTH = [
    (  0,   0),
	( 90,   0),
	( 91,  90),
	(269,  90),
	(270,   0),
	(359,   0),
]
