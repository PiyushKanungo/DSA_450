import numpy as np
import matplotlib.pyplot as plt
import random
import copy


class point:
    """A point in 2d"""

    def __init__(self, px, py):
        self.x = float(px)
        self.y = float(py)
        self.type = "point"

    def __repr__(self):
        return "point(" + str(self.x) + ", " + str(self.y) + ")"

    def rotate(self, theta):
        """Rotate the point around origin by angle theta"""
        rotmat = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])
        self.x, self.y = np.matmul(rotmat, [self.x, self.y])


def plotpoints(points):
    """plots points"""
    for k in range(len(points)):
        plt.plot(points[k].x, points[k].y, '.')
        plt.text(points[k].x, points[k].y, k)


class line:
    """A line in 2d"""

    def __init__(self, p1, p2):
        assert type(p1) is point and type(p2) is point,\
            "Input must be points"
        self.p1 = p1
        self.p2 = p2
        self.type = "line"

    def __repr__(self):
        return "line(" + str(self.p1) + ", " + str(self.p2) + ")"

    def length(self):
        return np.sqrt((self.p1.y - self.p2.y)**2 +
                       (self.p1.x - self.p2.x)**2)

    def rotate(self, theta):
        """Rotate the line around origin by angle theta"""
        self.p1.rotate(theta)
        self.p2.rotate(theta)


def plotlines(lines):
    """Plots line"""
    for k in range(len(lines)):
        plt.plot([lines[k].p1.x, lines[k].p2.x],
                 [lines[k].p1.y, lines[k].p2.y], '-')


class polygon:
    """A polygon in 2d"""

    def __init__(self, points):
        self.n = len(points)
        for p in points:
            pass
            assert type(p) is point,\
                "Edges must be lines"
        self.vertices = points
        self.edges = []
        for k in range(self.n - 1):
            self.edges.append(line(points[k], points[k+1]))
        self.edges.append(line(points[self.n - 1], points[0]))
        self.type = polygon

    def __repr__(self):
        ms = [p for p in self.vertices]
        return "polygon(" + str(ms) + ")"

    def rotate(self, th):
        """rotate a polygon"""
        for k in range(self.n):
            self.vertices[k].rotate(th)
            self.edges[k].rotate(th)


def plotpolygon(self):
    """Plots a polygon"""
    plotpoints(self.vertices)
    plotlines(self.edges)


def rotate(obj, th):
    """Rotate an object obj by angle th"""
    obj.rotate(th)


def isininterval(x0, x1, x2):
    """Numerically test if x0 is in the open interval x1 to x2"""
    eps = 1e-5
    if x1+eps < x0 < x2-eps or x2+eps < x0 < x1-eps:
        return True
    else:
        return False


def isintersect(line1, line2):
    """Test if two line segments l1 and l2 intersect.
    Lines are considered open so intersect at line edge
    counts as false."""
    l1 = copy.deepcopy(line1)
    l2 = copy.deepcopy(line2)
    if l1.length() == 0 or l2.length() == 0:
        return False
    while (l1.p1.x == l1.p2.x) or (l2.p1.x == l2.p2.x):
        # Rotate the whole world by 45deg to avoid dealing
        # with infinite a's which comes as a result of a
        # line being vertical
        rotate(l1, 45*np.pi/180)
        rotate(l2, 45*np.pi/180)
    # Solve for the line formula for both segments
    a1 = (l1.p2.y - l1.p1.y) / (l1.p2.x - l1.p1.x)
    a2 = (l2.p2.y - l2.p1.y) / (l2.p2.x - l2.p1.x)
    b1 = l1.p1.y - a1 * l1.p1.x
    b2 = l2.p1.y - a2 * l2.p1.x
    if a1 == a2:
        if b1 == b2:
            # This is a special case where two line segments
            # are on the same line
            if (isininterval(l2.p1.x, l1.p1.x, l1.p2.x) or
                isininterval(l2.p2.x, l1.p1.x, l1.p2.x) or
                isininterval(l1.p1.x, l2.p1.x, l2.p2.x) or
               isininterval(l1.p2.x, l2.p1.x, l2.p2.x)):
                return True
            else:
                return False
        else:
            return False
    else:
        # The intersection's x
        x = - (b2 - b1) / (a2 - a1)
        # If the intersection x is within the interval of each segment
        # then there is an intersection
        if (isininterval(x, l1.p1.x, l1.p2.x) and
           isininterval(x, l2.p1.x, l2.p2.x)):
            return True
        else:
            return False


def anyintersect(p):
    """Test if there is any intersect left at the polygon p
    or is it completly untangled"""
    assert type(p) is polygon, "Input must be polygon."
    return any([isintersect(p.edges[j], p.edges[k])
                for j in range(p.n)
                for k in range(p.n)])


def unwindpolygon(p, bailout = 50):
    """Greadily untangle a polygon by un crossing intersecting
    edges"""
    t = 0
    while anyintersect(p) and t < bailout:
        t += 1
        breakingflag = False
        for j in range(p.n):
            for k in range(p.n):
                if isintersect(p.edges[j], p.edges[k]):
                    if j < p.n - 1:
                        p12 = p.vertices[j+1]
                    else:
                        p12 = p.vertices[0]
                    if k < p.n - 1:
                        p22 = p.vertices[k+1]
                        p.vertices[k+1] = p12
                    else:
                        p22 = p.vertices[0]
                        p.vertices[0] = p12
                    if j < p.n - 1:
                        p.vertices[j+1] = p22
                    else:
                        p.vertices[0] = p22
                    breakingflag = True
                    break
            if breakingflag:
                break
        p.edges = polygon(p.vertices).edges


def shufflepolygon(p):
    """Randomely shuffle polygon vertices"""
    random.shuffle(p.vertices)
    p.edges = polygon(p.vertices).edges


# not used. Good for testing be rearranging manually a polygon
def rearrangepolygon(p, per):
    """p is polygon. per is a permutation of indices)"""
    p.vertices = [p.vertices[k] for k in per]
    p.edges = polygon(p.vertices).edges


def findbbox(pol):
    """Find the bounding box of a polygon"""
    maxx = minx = pol.vertices[0].x
    maxy = miny = pol.vertices[0].y
    for k in range(pol.n):
        minx = np.min([minx, pol.vertices[k].x])
        maxx = np.max([maxx, pol.vertices[k].x])
        miny = np.min([miny, pol.vertices[k].y])
        maxy = np.max([maxy, pol.vertices[k].y])
    return [minx, maxx, miny, maxy]


def isinside(pt, pol):
    """Test whether a point pt is inside polygon pol. The test
    is done by taking a line from pt to the edges of the bounding box
    and test if the number of intersections with the polygon edges is
    odd (pt is inside) or even (pt is outside)"""
    _, mx, _, _ = findbbox(pol)
    l = line(pt, point(mx + 1, pt.y + 1))
    # line is not completly horizontal to avoid specuial cases in "isintersect"
    res = sum([isintersect(l, pol.edges[k]) for k in range(pol.n)])
    if not res % 2:
        return False
    else:
        return True


def findarea(p, n=1000):
    """Finds the area by monte-carlo. Drop points and count the fraction
    that is inside the polygon"""
    minx, maxx, miny, maxy = findbbox(p)
    area = (maxx - minx) * (maxy - miny)
    fra = 0
    for _ in range(n):
        ptx = minx + (maxx - minx)*np.random.rand()
        pty = miny + (maxy - miny)*np.random.rand()
        pt = point(ptx, pty)
        fra += isinside(pt, p)
    return area * fra / n


def randomtestpolygon(p, t=20, acc = 200):
    """Repeatedly shuffle and untangle, keeping record of the largest
    area result, for t trials. Some plottings are added to
    show what's going on."""
    hasbeenbestfor = 0
    bestscore = findarea(p, acc)
    pbest = copy.deepcopy(p)
    plt.figure()
    plt.subplot(121)
    plotpolygon(p)
    plt.axis('square')
    plt.title("best area:" + str(bestscore))
    print("trial, largest area, current area")
    # Look for the configuration that can hold against competition
    # for t trials
    while hasbeenbestfor < t:
        shufflepolygon(p)
        unwindpolygon(p)
        currentscore = findarea(p, acc)
        print(hasbeenbestfor, bestscore, currentscore)
        plt.subplot(122)
        plt.cla()
        plotpolygon(p)
        plt.axis('square')
        plt.title("current area:" + str(currentscore))
        plt.draw()
        plt.pause(0.1)
        if currentscore > bestscore:
            bestscore = currentscore
            pbest = copy.deepcopy(p)
            hasbeenbestfor = 0
            plt.subplot(121)
            plt.cla()
            plotpolygon(p)
            plt.axis('square')
            plt.title("best area:" + str(bestscore))
        else:
            hasbeenbestfor += 1
    p.vertices = pbest.vertices
    p.edges = pbest.edges


if __name__ == '__main__':
    plt.close('all')
    n = 7  # number of vertices
    # This is just random vertices generator. The details are not important
    # This way ensures that a nice untangled polygon is likely
    r = 10+100*np.random.rand(n,)
    th = [(k + 10*n/360*np.random.randn())*np.pi/180
          for k in np.linspace(0, 360, n)]
    vertices = [point(r_*np.cos(th_), r_*np.sin(th_))
                for r_, th_ in zip(r, th)]
    p = polygon(vertices)
    # but let's start with something hard
    shufflepolygon(p)
    randomtestpolygon(p, 30, 1000)
