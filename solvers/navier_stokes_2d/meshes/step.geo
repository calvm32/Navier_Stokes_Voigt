// ----------
// Parameters
// ----------

L = 4.0;
H = 1.0;

step_width = 0.1;	    // step width
step_height = 0.1;	    // step distance from y=0
step_distance = 0.5;	// step distance from x=0

lc = 0.015;		// mesh size

// ------------------
// Points (CCW order)
// ------------------

// Bottom wall with step
Point(1) = {0, 0, 0, lc};					                    // bottom left
Point(2) = {step_distance, 0, 0, lc};				            // step start
Point(3) = {step_distance, step_height, 0, lc};			        // step left top
Point(4) = {step_distance + step_width, step_height, 0, lc};	// step right top
Point(5) = {step_distance + step_width, 0, 0, lc};		        // step end
Point(6) = {L, 0, 0, lc};					                    // bottom right

// Top wall
Point(7) = {L, H, 0, lc};	// top-right
Point(8) = {0, H, 0, lc};   // top-left

// -----
// Lines
// -----

// Bottom wall with step
Line(1) = {1,2};  // bottom horizontal left from origin
Line(2) = {2,3};  // vertical up step
Line(3) = {3,4};  // horizontal across step
Line(4) = {4,5};  // vertical down step
Line(5) = {5,6};  // bottom horizontal right to end
Line(6) = {6,7};  // right vertical wall
Line(7) = {7,8};  // top wall
Line(8) = {8,1};  // left vertical wall

// -------------------
// Line loop & surface
// -------------------

Line Loop(1) = {1,2,3,4,5,6,7,8};  // all lines forming perimeter
Plane Surface(1) = {1};

// --------------
// Physical lines
// --------------

Physical Line("Left") = {8};		    // left wall (id 1)
Physical Line("Right") = {6};		    // right wall (id 2)
Physical Line("Bottom") = {1,2,3,4,5};	// bottom including step (id 3)
Physical Line("Top") = {7};		        // top wall (id 4)

// ----------------
// Physical surface
// ----------------

Physical Surface("Fluid") = {1};

// ------------
// Mesh control
// ------------

// Absolutely forbid quads
Mesh.RecombineAll = 0;
Mesh.Recombine3DAll = 0;

// Force triangle-only meshing
Mesh.Algorithm = 5; // Delaunay = TRIANGLES ONLY

// Generate 2D mesh
Mesh 2;