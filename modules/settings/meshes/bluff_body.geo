// INFO: 
// V Total DoFs: 
// W Total DoFs: 

// ----------
// Parameters
// ----------

L = 40.0;   // length
H = 10.0;   // height

lc = 2.0;   // mesh size (far field)
lc2 = 1.0;  // mesh size (airfoil surface — finer)

chord = 1.0;
x_offset = 8.0;   // leading edge x position
y_offset = H/2;   // vertically centered

// ------------------
// Points (CCW order)
// ------------------

Point(1) = {0, 0, 0, lc};	// bottom left
Point(2) = {L, 0, 0, lc};	// bottom right
Point(3) = {L, H, 0, lc};	// top-right
Point(4) = {0, H, 0, lc};   // top-left

// -----
// Lines
// -----

// Bottom wall
Line(1) = {1,2};  // bottom wall
Line(2) = {2,3};  // right wall
Line(3) = {3,4};  // top wall
Line(4) = {4,1};  // left wall

// -------------------
// Line loop & surface
// -------------------

Line Loop(1) = {1,2,3,4};  // all lines forming perimeter
Plane Surface(1) = {1};

// --------------
// Physical lines
// --------------

Physical Line("Left") = {4};		    // left wall (id 1)
Physical Line("Right") = {2};		    // right wall (id 2)
Physical Line("Bottom") = {1};	        // bottom wall (id 3)
Physical Line("Top") = {3};		        // top wall (id 4)

// ----------------
// bluff body lines
// ----------------

radius = chord/2;

xc = x_offset + radius;
yc = y_offset;

// Circle points
Point(101) = {xc + radius, yc, 0, lc2};
Point(102) = {xc, yc + radius, 0, lc2};
Point(103) = {xc - radius, yc, 0, lc2};
Point(104) = {xc, yc - radius, 0, lc2};
Point(105) = {xc, yc, 0, lc2};

// Circle arcs (counterclockwise)
Circle(1001) = {101,105,102};
Circle(1002) = {102,105,103};
Circle(1003) = {103,105,104};
Circle(1004) = {104,105,101};

// ----------------
// Physical surface
// ----------------

Spline(1001)={101 : 185};
Spline(1002)={185 : 209};
Spline(1003)={209 : 292, 101};

Line Loop(2001)={1,2,3,4};
Line Loop(2002) = {1001,1002,1003,1004};
Plane Surface(3001)={2001,2002};

Physical Surface("Fluid") = {3001};
Physical Line("Airfoil") = {1001, 1002, 1003};

// ------------
// Mesh control
// ------------

// forbid quads
Mesh.RecombineAll = 0;
Mesh.Recombine3DAll = 0;

// Force triangle-only meshing
Mesh.Algorithm = 5; // Delaunay = TRIANGLES ONLY

// Generate 2D mesh
Mesh 2;
