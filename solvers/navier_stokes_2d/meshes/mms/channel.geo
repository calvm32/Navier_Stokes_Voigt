// INFO: 
// V Total DoFs: 
// W Total DoFs: 

// ----------
// Parameters
// ----------

L = 40.0;   // length
H = 10.0;   // height

lc = 1.0;	// mesh size

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

// Bottom wall with step
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
// Physical surface
// ----------------

Physical Surface("Fluid") = {1};

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