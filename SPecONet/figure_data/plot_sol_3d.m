sigma = 10;
load("./data/fig5/3d_force_data/3d_force" + num2str(sigma) + "sigma.mat");
load("./data/fig5/3d_force_data/xx.mat","x");

xslice = x(end-5);
yslice = x(end-5);
zslice = x(5);

[x, y, z] = meshgrid(x,x,x);

jj=1;
zz=2; % timestep = [0.25, 0.5, 0.75, 1.0]
u=squeeze(bar(jj,1,zz,:,:,:));
v=squeeze(bar(jj,2,zz,:,:,:));
w=squeeze(bar(jj,3,zz,:,:,:));

speed = sqrt(u.^2 + v.^2 + w.^2);

% Start Figure
figure('Position', [500, 300, 300, 250])
hold on;

% Slice 
hsurfaces = slice(x, y, z, speed, xslice, yslice, zslice);
set(hsurfaces, 'FaceColor', 'interp', 'EdgeColor', 'none')

% Streamlines
sx1 = xslice; 
sy1 = yslice;
sz1 = zslice;
density = 0.15;
l1 = streamslice(x, y, z, u, v, w, sx1, sy1, sz1, density);
set(l1, 'Color', [1 1 1 0.7], 'LineWidth', 1.5)


axis equal tight
view(3)

xticks([-1 0 1]);
yticks([-1 0 1]);
zticks([-1 0 1]);
xticklabels({'','x',''});
yticklabels({'','y',''});
zticklabels({'','z',''});

grid on;
ax = gca;             
ax.FontSize = 14;

camlight 
lighting gouraud 


load('viridis_cmap.mat');
colormap(viridi);
c = colorbar;  
c.Position(3) = 0.02; 
c.Position(4) = 0.6; 
c.Position(2) = 0.2;   
c.Position(1) = 0.85;