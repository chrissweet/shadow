import sys
import numpy as np
from math import *
from Polygon import *
from Polygon.Utils import pointList
from itertools import permutations
from random import sample

#flag if OpenGL available on this system
Opengl_available = True

try:
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    from OpenGL.GL import *
    print "OpenGL available."
except ImportError:
    opengl_available = False
    print "OpenGL NOT available!"

########################################################################################################################
# Shadow class to calculate shadow geometries for a set of surfaces on a per surface basis
# Author: Chris Sweet, Center for Research Computing 14th February 2015
# Notice:
# Licensee (Greenscale Project at Notre Dame) acknowledges that this is only a limited nonexclusive license
# for this software.
# Licensor (The Center for Research Computing) is and remains the owner of all titles, rights, and interests
# in this Software.
#
# Assumptions:
# 1) The outer surface is ordered counter-clockwise
# 2) Only outer surfaces are considered for shadowing
#
# Main functions:
# 1) add_surface(numpy array of 3 or more point), Adds a surface to the class, at least two required
# 2) find_shadows(), finds shadows on a per surface basis, does a full N^2 search
# 3) get_shadow_ratio(surface number, numpy array defining geometry), get the shadowed ratio of an object on a surface
# 4) visualize(), launches OpenGL visualization of the structure with shadows
# 5) set_sun(numpy array with azimuth and tilt of the sun), sets the sun position
# 6) __init__(boolean for OpenGL available), instantiates with a flag to define if OpenGL is available on the system
# 7) get_surface_shadow_ratio(surface number), gets the shadowed ratio for a surface
#
########################################################################################################################
class Shadow:
    #~~~~Variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~OpenGL specific~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Opengl_status = True

    ESCAPE = as_8_bit('\033')

    PROMPT = ("Press key  'r' to start/stop rotation",
              "Press keys 't'/'y' to change Sun tilt",
              "Press keys 'a'/'s' to change Sun azimuth",
              "Press keys '+'/'-' to zoom",
              "Press key  'q' to attach Sun to new surface",
              "Press ESCAPE to exit.")

    name = 'Shadow_test'

    #allow rotation
    rotate_model = True
    rotate_count = 0

    #set scale to zoom
    scale = 1.0

    #surface for sun vector
    sun_vector_attach = 0

    #holds colors for surfaces
    colors = []

    #~~~~Shadow specific~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #shadow variables for class
    surfaces = []
    shadows = []
    combined_shadows = []
    normals = []
    sun_vector = np.array([])
    sun = np.array([])
    model_rotation = 0

    #~~~~OpenGL section of the code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #start opengl
    def visualize(self):
        #test data and OpenGL available
        if self.Opengl_status and len(self.surfaces) >= 2 and self.sun.shape[0] > 0:
            #generate colors
            num_surfaces = len(self.surfaces)
            color_lvl = 8
            rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
            self.colors = sample(rgb,num_surfaces)

            glutInit(sys.argv)
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(600,600)
            glutCreateWindow(self.name)

            glClearColor(0.,0.,0.,1.)
            glShadeModel(GL_SMOOTH)

            #lighting
            #glEnable(GL_CULL_FACE)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            lightZeroPosition = [10.,4.,10.,1.]
            lightZeroColor = [0.8,1.0,0.8,1.0] #green tinged
            glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
            glLightfv(GL_LIGHT0, GL_AMBIENT, lightZeroColor)
            glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
            glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
            glEnable(GL_LIGHT0)

            #initialize functions
            glutKeyboardFunc(self.keyboard)
            glutDisplayFunc(self.display)
            glutReshapeFunc(self.reshape)

            glutMainLoop()
        else:
            print "No Sun or surface data or OpenGL libraries!"
        return

    #set shape if window re-sized
    def reshape(self, width,height):
        glViewport(0, 0, width, height)

    #keyboard interaction
    def keyboard(self, key, x_coord, y_coord):
        if key == self.ESCAPE:
            sys.exit()

        #auto-rotate model?
        if key == 'r':
            if self.rotate_model == True:
                self.rotate_model = False
            else:
                self.rotate_model = True

        #move sun azimuth?
        if key == 'a':
            self.sun[0] = self.sun[0] + 0.1
            #try full n^2 shadow find
            self.find_shadows()

        if key == 's':
            self.sun[0] = self.sun[0] - 0.1
            #try full n^2 shadow find
            self.find_shadows()

        #move sun tilt?
        if key == 't':
            self.sun[1] = self.sun[1] + 0.1
            #try full n^2 shadow find
            self.find_shadows()
        if key == 'y':
            self.sun[1] = self.sun[1] - 0.1
            #try full n^2 shadow find
            self.find_shadows()

        #zoom?
        if key == '=' or key == '+':
            self.scale = self.scale * 1.05

        if key == '-':
            self.scale = self.scale / 1.05

        #attach sun vector to surface
        if key == 'q':
            self.sun_vector_attach = self.sun_vector_attach + 1

    def display(self):
        #viewport
        w = float(glutGet(GLUT_WINDOW_WIDTH))
        h = float(glutGet(GLUT_WINDOW_HEIGHT))
        glViewport(0, 0, int(w), int(h))

        #setup view
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.,w/h,1.,40.)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,20,10,
                  0,0,0,
                  0,-1,0)

        #draw display
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        color = [1.0,0.,0.,1.]
        glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)

        #rotate the view
        if self.rotate_model == True:
            self.rotate_count = self.rotate_count + 1
            if self.rotate_count > 360:
                self.rotate_count = 0

        glRotatef(self.rotate_count,0,0,1)
        #glutSolidSphere(2,20,20)

        #scale
        glScale(self.scale,self.scale,self.scale)

        #find com
        com = np.array([0,0,0])
        num_surfaces = len(self.surfaces)
        num_points = 0
        for i in range(0, num_surfaces):
            sizesu = self.surfaces[i].shape[0]
            num_points = num_points + sizesu
            for j in range(0,sizesu):
                com = np.add(com, self.surfaces[i][j])

        com = np.multiply(com, 1.0 / num_points)

        #translate to com
        glTranslatef(-com[0], -com[1], -com[2])

        #draw the surfaces
        for i in range(0, num_surfaces):
            #set color
            #color = [1.0,0.,0.,1.]
            glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,self.colors[i])

            #draw surface
            glBegin(GL_POLYGON) #starts drawing of points

            num_pnt = self.surfaces[i].shape[0]
            for j in range(0, num_pnt):
                glVertex3f(self.surfaces[i][j][0],self.surfaces[i][j][1],self.surfaces[i][j][2])

            glEnd() #end drawing of points

        #draw cylinder for sun ray
        #get index to attach
        sindex = self.sun_vector_attach % num_surfaces

        #color = [1.0,1.0,1.0,1.]
        glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,self.colors[sindex])

        s_start = self.surfaces[sindex][0]
        s_end = np.add(self.sun_vector, s_start)
        z = np.array([0,0,1])

        #get angle
        ang = acos(np.dot(z, self.sun_vector))

        #get cross
        cross = np.cross(z, self.sun_vector)

        #
        glPushMatrix()

        #move
        glTranslatef(s_start[0],s_start[1],s_start[2])
        glRotatef(ang * 180.0/pi, cross[0], cross[1], cross[2])

        #draw
        glutSolidSphere(0.1,10,10)
        quadratic = gluNewQuadric()
        gluCylinder(quadratic, 0.05, 0.05, 1, 10, 10)      # to draw the lateral parts of the cylinder;

        #move
        glTranslatef(0.0,0.0,1.0)

        #draw
        gluCylinder(quadratic, 0.1, 0.001, 0.5, 10, 10)      # to draw the lateral parts of the cylinder;

        glPopMatrix()
        #end sun vector

        #shadow color
        color = [0.1,0.1,0.1,1.]
        glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)

        #draw shadow for each surface
        num_shadows = len(self.shadows)
        for i in range(0, num_shadows):
            num_shadows = len(self.shadows[i])
            for j in range(0, num_shadows):
                numpoints = self.shadows[i][j].shape[0]
                if numpoints > 0:
                    glBegin(GL_POLYGON) #starts drawing of points

                    for k in range(0,numpoints):
                        shadowepsilon = np.add(self.shadows[i][j][k], np.multiply(self.normals[i],0.01))
                        glVertex3f(shadowepsilon[0],shadowepsilon[1],shadowepsilon[2])

                    glEnd() #end drawing of points

        #end of 3D draw
        glPopMatrix()

        #draw compass
        glPushMatrix()
        color = [1.0,1.0,1.0,1.]
        glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)

        #move
        glTranslatef(-5.0,5.0,-3.5)
        glRotatef(self.rotate_count + (self.model_rotation * 180.0 / pi),0,0,1)
        glRotatef(-90.0, 1.0, 0.0, 0.0)

        #draw
        glutSolidSphere(0.1,10,10)
        quadratic = gluNewQuadric()
        gluCylinder(quadratic, 0.05, 0.05, 1, 10, 10)      # to draw the lateral parts of the cylinder;

        #move
        glTranslatef(0.0,0.0,1.0)

        #draw
        quadratic = gluNewQuadric()
        gluCylinder(quadratic, 0.1, 0.001, 0.5, 10, 10)      # to draw the lateral parts of the cylinder;

        glPopMatrix()

        #end of compass

        #text printout
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 0.5, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(-1.0, 1.0, 0.0)
        cscale = 1.0/w
        glScale(cscale, -cscale*w/h, 1.0)
        cy = 25.0
        for s in self.PROMPT:
            glRasterPos(40.0, cy)
            cy += 30.0
            for c in s:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

        #print sun data
        sunstring = "Sun azimuth, tilt: "+str(self.sun[0])+", "+str(self.sun[1])
        glRasterPos(40.0, cy)
        cy += 30.0
        for c in sunstring:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

        #print shadow percent
        ratio = self.get_surface_shadow_ratio(sindex)

        glRasterPos(40.0, cy)
        areastring = "Shadowed area for surface "+str(sindex+1)+" is "+'%.2f' % (ratio*100.0)
        for c in areastring:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

        #model rotation
        cy += 30.0
        glRasterPos(40.0, cy)
        rotatestring = "Model rotation "+'%.2f radians.' % self.model_rotation
        for c in rotatestring:
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

        glEnable(GL_LIGHTING)

        #show it all
        glutSwapBuffers()

        #redraw next frame
        glutPostRedisplay()
        return

    #~~~~Routines for shadow calculation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #find ratio of shadow to area for a surface
    def get_surface_shadow_ratio(self, sindex):
        if len(self.combined_shadows) > sindex:
            #find axes on first surface
            xi, yi = self.find_axes_in_surface_plane(self.surfaces[sindex], self.normals[sindex])

            #find this surfaces points on plane
            my2d = self.find_2D_points_in_plane(self.surfaces[sindex], xi, yi, self.surfaces[sindex][0])

            surface_area = Polygon(my2d).area()

            shadow_area = self.combined_shadows[sindex].area()

            return shadow_area / surface_area
        else:
            return 0.0

    #find ratio of shadow to area for any geometry on a surface (i.e. a window)
    def get_shadow_ratio(self, sindex, geometry):
        if len(self.combined_shadows) > sindex:
            #find axes on first surface
            xi, yi = self.find_axes_in_surface_plane(self.surfaces[sindex], self.normals[sindex])

            #find this surfaces points on plane
            my2d = self.find_2D_points_in_plane(geometry, xi, yi, self.surfaces[sindex][0])

            myPoly = Polygon(my2d)
            surface_area = myPoly.area()

            #get intersection of combined shadow with the geometry
            shadow_area = (self.combined_shadows[sindex] & myPoly).area()

            return shadow_area / surface_area
        else:
            return 0.0

    #find vector normal to the surface
    def calculate_surface_vector(self, surface):
        #find two vectors on surface, wlog pick 1, 2, 3
        v1 = np.subtract(surface[1], surface[0])
        v2 = np.subtract(surface[1], surface[2])

        #find cross product to get normal vector, then normalize it
        n = np.cross(v2, v1)
        n = n / np.linalg.norm(n)
        return n

    #find actual sun vector from azimuth, tilt
    def calculate_sun_vector(self, sun):
        #get components, subtract model_rotation from it
        azimuth = sun[0] - self.model_rotation
        tilt = sun[1]
        #sun "direction cosines", changed cos(tilt) to sin(tilt) in first term
        sv = np.array([sin(azimuth)*sin(tilt), cos(azimuth)*sin(tilt), cos(tilt)])

        #print np.linalg.norm(sv)
        #sv = sv / np.linalg.norm(sv)

        return sv

    #generate x-y axes for this surface plane. Used to find 2D representation of points
    def find_axes_in_surface_plane(self, surf, surf_normal):
        #Create basis set in 2D, surface_normal is n, surfaces[i][0] is r0 surface points r satisfy n(r-r0)=0
        #wlog pick first two points as the x axis and normalize
        x_axis = np.subtract(surf[1], surf[0])
        x_axis = x_axis / np.linalg.norm(x_axis)

        #rotate by pi/2 around n to form the y axis, use the general method below:
        #rotate point around view vector to get orthogonal
        #From http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ArbitraryAxisRotation.html
        #a(v^2+w^2)+u(-bv-cw+ux+vy+wz)+(-cv+wy+vz)
        #b(u^2+w^2)+v(-au-cw+ux+vy+wz)+(cu-aw+wx-uz)
        #c(u^2+v^2)+w(-au-bv+ux+vy+wz)+(-bu+av-vx+uy)
        #x,y,z point, a,b,c center, u,v,w vector(=a,b,c),
        x = x_axis[0]
        y = x_axis[1]
        z = x_axis[2]
        a = surf_normal[0]
        b = surf_normal[1]
        c = surf_normal[2]
        u = a
        v = b
        w = c

        dotp1 = u*x+v*y+w*z
        xd = a*(v*v+w*w)+u*(-b*v-c*w+dotp1)+(-c*v+b*w-w*y+v*z)
        yd = b*(u*u+w*w)+v*(-a*u-c*w+dotp1)+(c*u-a*w+w*x-u*z)
        zd = c*(u*u+v*v)+w*(-a*u-b*v+dotp1)+(-b*u+a*v-v*x+u*y)

        y_axis = np.array([xd, yd, zd])
        y_axis = y_axis / np.linalg.norm(y_axis)

        #return axes
        return x_axis, y_axis

    #find points projected onto this surface by the sun
    def find_points_on_surface_plane(self, my_surface, shadowed_surface, shadowed_surface_normal, sun):
        #find line intersectionP(s)= P0 + su, where P0 is the point to project, s is parametric and u is the sun vector
        #Then sI = n.(V0-P0)/n.u, where a,b,c are the n components
        #TODO we can check for -ve s in case that it does not shadow

        #set flag for all behind surface (all sI +ve)
        behind_surface = True
        #find number of points
        number_my_points = my_surface.shape[0]
        output_points = np.empty([number_my_points,3])
        for k in range(0,number_my_points):
            si = np.dot(shadowed_surface_normal, np.subtract(shadowed_surface[0], my_surface[k])) / np.dot(shadowed_surface_normal, sun)

            #test if in front of wall as we need to set flag (0 doesnt count)
            if si < 0:
                behind_surface = False
            output_points[k] = np.add(my_surface[k], np.multiply(sun, si))

        #if all points behind surface send empty array
        if behind_surface:
            output_points = np.array([])

        #return
        return output_points

    #find 2D representation of 3D points in the plane
    def find_2D_points_in_plane(self, points_on_plane, x_axis, y_axis, origin):
        number_points = points_on_plane.shape[0]
        output2D = np.empty([number_points,2])
        for i in range(0,number_points):
            point_from_origin = np.subtract(points_on_plane[i], origin)
            output2D[i][0] = np.dot(point_from_origin, x_axis)
            output2D[i][1] = np.dot(point_from_origin, y_axis)

        #return 2D points
        return output2D

    #Do inverse of above to recover 3D from 2D points
    def project_2D_to_plane(self, input_points, x_axis, y_axis, origin):
        number_points = input_points.shape[0]
        output3D = np.empty([number_points,3])
        for i in range(0,number_points):
            output3D[i] = np.add(np.add(np.multiply(x_axis, input_points[i][0]), np.multiply(y_axis, input_points[i][1])), origin)

        #return the 3D
        return output3D

    #do intersection of polygons
    def poly_intersection(self, first2D, second2D):
        #convert 2D numpy arrays to polygons for polygon library
        psurf = Polygon(first2D)

        pshad = Polygon(second2D)

        #get intersection
        pint = psurf & pshad

        if bool(pint):
            #back to numpy array
            pintarray = np.array(pointList(pint))
        else:
            pintarray = np.array([])

        #return both numpy and polygon representation
        return pintarray, pint

    #find n^2 shadows
    def find_shadows(self):
        #test data available
        if len(self.surfaces) < 2 or self.sun.shape[0] == 0:
            print "No Sun or surface data!"
            return False

        #list of lists to save
        self.shadows = []

        #combined poly list
        self.combined_shadows = []

        #find number of surfaces
        num_surfaces = len(self.surfaces)

        #get sun vector
        self.sun_vector = self.calculate_sun_vector(self.sun)

        #get normals for surfaces
        self.normals = []

        for i in range(0,num_surfaces):
            self.normals.append(self.calculate_surface_vector(self.surfaces[i]))

        #loop over surfaces to look for shadows
        for i in range(0,num_surfaces):
            #check I can see the sun
            cosine_angle = np.dot(self.normals[i], self.sun_vector)

            #if not get next surface
            if cosine_angle <= 0:
                self.shadows.append([])
                self.combined_shadows.append(Polygon())
                continue

            #****here if sunny!****
            #find axes on first surface
            xi, yi = self.find_axes_in_surface_plane(self.surfaces[i], self.normals[i])

            #find this surfaces points on plane
            my2di = self.find_2D_points_in_plane(self.surfaces[i], xi, yi, self.surfaces[i][0])

            #create list
            shadowlist = []

            #create combined polygon
            combined_shadow = Polygon()

            #loop over other surfaces to look for shadows
            for j in range(0,num_surfaces):
                if i == j:
                    continue    #dont compare to self but must look at all others as shadow not a bijection

                #check second surface can see the sun
                ss_cosine_angle = np.dot(self.normals[j], self.sun_vector)
                #if not get next surface
                if ss_cosine_angle <= 0:
                    shadowlist.append(np.array([]))
                    continue

                #potential for shadow if here
                #find shadow points on first surface
                #first in 3D
                ponsj = self.find_points_on_surface_plane(self.surfaces[j], self.surfaces[i], self.normals[i], self.sun_vector)

                #then 2D in the plane of the first surface
                pons2d = self.find_2D_points_in_plane(ponsj, xi, yi, self.surfaces[i][0])

                #do intersection of polygons
                pintarray, poly = self.poly_intersection(my2di, pons2d)

                if pintarray.shape[0] > 0:
                    #then project final back to 3D
                    shadowi = self.project_2D_to_plane(pintarray, xi, yi, self.surfaces[i][0])

                    #and accumulate polygon
                    combined_shadow = combined_shadow | poly
                else:
                    shadowi = np.array([])

                #save one for each j
                shadowlist.append(shadowi)

            #after loop append shadow list to shadows (list of lists of numpy arrays)
            self.shadows.append(shadowlist)

            #print combined_shadow.area(),i
            #also append combined polys
            self.combined_shadows.append(combined_shadow)

        #return status
        return True

    #Set model rotation, use Revit/Energy Plus convention of positive->clockwise rotation
    #This version uses a surface and its reported azimuth in degrees
    def set_model_rotation_with_azimuth(self, surfindex, surfazimuth):
        if surfindex >= len(self.surfaces):
            return False
        else:
            snormal = self.calculate_surface_vector(self.surfaces[surfindex])
            surfangle = atan2(snormal[0],snormal[1])
            self.model_rotation = (surfazimuth * pi) / 180.0 - surfangle
            return True

    #Set model rotation, use Revit/Energy Plus convention of positive->clockwise rotation
    #Direct set method
    def set_model_rotation(self, rot):
        self.model_rotation = rot

    #set sun
    def set_sun(self, sunin):
        self.sun = sunin
        #get sun vector
        self.sun_vector = self.calculate_sun_vector(self.sun)

    #add surface
    def add_surface(self, surface):
        self.surfaces.append(surface)

    #set opengl status on instantiate
    def __init__(self, opengl_stat):
        self.Opengl_status = opengl_stat

    #~~~~End of Shadow class~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~main program~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    #input data
    #surfaces. Start with East facing wall with overhang
    su2 = np.array([[2.0368525176, -0.6640470336000001, 0.0], [2.0368525176, 5.835178068, 0.0], [2.0368525176, 5.835178068, 3.2512001016000003], [2.0368525176, -0.6640470336000001, 3.2512001016000003]])
    shadow1 = np.array([[2.0368525176, -0.6640470336000001, 3.2512001016000003], [3.0368525176, -0.6640470336000001, 3.2512001016000003], [3.0368525176, 5.835178068, 3.2512001016000003], [2.0368525176, 5.835178068, 3.2512001016000003]])

    #others
    su_1 = np.array([[-4.462372584000001, 5.835178068, 0.0], [-4.462372584000001, 5.835178068, 3.2512001016000003], [2.0368525176, 5.835178068, 3.2512001016000003], [2.0368525176, 5.835178068, 0.0]])
    #extended bottom for second shadow, second point 2.0368525176 -> 2.5368525176
    su_3 = np.array([[-4.462372584000001, -0.6640470336000001, 0.0], [2.5368525176, -0.6640470336000001, 0.0], [2.0368525176, -0.6640470336000001, 3.2512001016000003], [-4.462372584000001, -0.6640470336000001, 3.2512001016000003]])
    su_4 = np.array([[-4.462372584000001, -0.6640470336000001, 0.0], [-4.462372584000001, -0.6640470336000001, 3.2512001016000003], [-4.462372584000001, 5.835178068, 3.2512001016000003], [-4.462372584000001, 5.835178068, 0.0]])
    su_5 = np.array([[-4.462372584000001, -0.6640470336000001, 3.2512001016000003], [2.0368525176, -0.6640470336000001, 3.2512001016000003], [2.0368525176, 5.835178068, 3.2512001016000003], [-4.462372584000001, 5.835178068, 3.2512001016000003]])
    su_6 = np.array([[-4.462372584000001, -0.6640470336000001, 0.0], [-4.462372584000001, 5.835178068, 0.0], [2.0368525176, 5.835178068, 0.0], [2.0368525176, -0.6640470336000001, 0.0]])

    #new test surface
    ns = np.array([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, -1.0, 5.0], [0.66, -1.0, 5.0], [0.66, -1.0, 4.9], [0.33, -1.0, 4.9], [0.33, -1.0, 5.0], [0.0, -1.0, 5.0]])

    #create class, pass opengl availability
    myshadow = Shadow(Opengl_available)

    #set model rotation, positive is clockwise
    myshadow.set_model_rotation(0.0)

    #Sun azimuth/tilt
    myshadow.set_sun(np.array([1.7944, 0.9521]))

    #put surfaces into list
    myshadow.add_surface(su2)
    myshadow.add_surface(shadow1)

    #others
    myshadow.add_surface(su_1)
    myshadow.add_surface(su_3)
    myshadow.add_surface(su_4)
    myshadow.add_surface(su_5)
    myshadow.add_surface(su_6)
    myshadow.add_surface(ns)

    #set model rotation after setting surfaces, first argument surface index, second reported azimuth in degrees
    myshadow.set_model_rotation_with_azimuth(2,2.0)

    #use full n^2 shadow find
    myshadow.find_shadows()

    #test geometry ratio
    print "Shadow for 0 ", myshadow.get_shadow_ratio(0, su2)

    #print myshadow.normals[2], atan2(myshadow.normals[2][0],myshadow.normals[2][1])

    #do OpenGL loop
    myshadow.visualize()
