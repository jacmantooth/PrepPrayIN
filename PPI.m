xa = 0; xb = 300; ya = 0; yb = 300;
nx =  600; ny = 600; 
x_f = linspace(xa,xb,nx); hx = x_f(2)-x_f(1);
y_f = linspace(ya,yb,nx); hy = y_f(2)-y_f(1);
dx = .5;
dt =1/384 ;
alpha = .5;
beta = .01;

gamma = .5; 
delta = 1; 
radius = 75;

d = (dt)/(dx^2);
e = (1/384 * delta)/(.5^2);


center_x = 150;
center_y = 150;
[X, Y] = meshgrid(x_f, y_f);
distancefromcenter = sqrt((X - center_x).^2 + (Y - center_y).^2);


u_0 = zeros(size(distancefromcenter));
u_0(distancefromcenter <= radius) = 0.2;
u_0 = u_0(:);
v_0 = zeros(size(distancefromcenter));
v_0(distancefromcenter <= radius) = 0.2;
v_0 = v_0(:);


totalv = zeros(1, 11001);
totalu = zeros(1, 11001);
totalu(1) = sum(u_0);
totalv(1) = sum(v_0);
for c = 1:11000

    newu = getnewu(nx,ny,d,dt,beta,alpha,u_0,v_0);
    newv = getnewv(nx,ny,gamma,e,dt,alpha,u_0,v_0);
    u_0 = newu;
    v_0 = newv;
    totalu(c+1) = sum(u_0);
    totalv(c+1) = sum(v_0);
    if mod(c,1000) == 0 
        newv = reshape(newv,[600,600]);
        newu = reshape(newu,[600,600]);
        subplot(2, 2, 1);
        mesh(newv);
        colorbar;
        title('Mesh of newv'); 
        
        % Create the second subplot
        subplot(2, 2, 2); 
        mesh(newu); 
        colorbar;
        title('Mesh of newu'); 
        
        subplot(2, 2, [3,4]);
        plot(totalu(1:c), totalv(1:c), 'r-o');
        xlabel('Total u');
        ylabel('Total v');
        title('Population of u vs. Population of v Over Time');
        grid on;
        drawnow;

    end 
end


figure; % Create a new figure

% Create the first subplot
subplot(1, 2, 1); 
mesh(newv);
colorbar;
title('Mesh of newv'); 

% Create the second subplot
subplot(1, 2, 2); 
mesh(newu); 
colorbar;
title('Mesh of newu'); 

function newu = getnewu(nx,ny,d,dt,beta,alpha,u0,v0)
F = u0- dt*(u0.*v0)./(u0-alpha);
mid = 1 +4*d - dt*beta*(1-u0);

Ix = speye(nx);
Iy = speye(ny);

ex = ones(nx,1);
exupper = ones(nx,1);
exlower  = ones(nx,1);
exupper(2) = 2;
exlower(end-1) = 2;
D2X = spdiags([exlower*-d 2*ex exupper*-d],[-1 0 1],nx,nx);
D2Y = spdiags([exlower*-d 2*ex exupper*-d],[-1 0 1],ny,ny);

M.D2YY= kron(D2Y,Iy);
M.D2XX = kron(Ix,D2X);
M.D2XY = M.D2XX + M.D2YY;
M.D2XY =M.D2XY;
M.D2XY = M.D2XY + spdiags(mid - diag(M.D2XY), 0, nx*ny, nx*ny); 

newu = M.D2XY\F;

end 

function newv = getnewv(nx,ny,gamma,e,dt,alpha,u0,v0)
F = v0+ dt*(u0.*v0)./(u0-alpha);
mid = (1 +4*e +dt*gamma)*ones(nx,1);
Ix = speye(nx);
Iy = speye(ny);

exupper = ones(nx,1);
exlower  = ones(nx,1);
exupper(2) = 2;
exlower(end-1) = 2;
D2X = spdiags([-e*exlower mid -e*exupper],[-1 0 1],nx,nx);
D2Y = spdiags([-e*exlower 0*exupper -e*exupper],[-1 0 1],ny,ny);

M.D2YY= kron(D2Y,Iy);
M.D2XX = kron(Ix,D2X);
M.D2XY = M.D2XX + M.D2YY;

newv = M.D2XY\F;

end 
