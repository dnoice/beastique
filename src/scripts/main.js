// document.addEventListener('DOMContentLoaded', function() {
//     console.log('Beastique application initialized');
    
// });

// async function fetchAnimalData() {
//     try {
//         const response = await fetch('/src/data/animals.json');
//         const data = await response.json();
//         return data;
//     } catch (error) {
//         console.error('Error fetching animal data:', error);
//         return [];
//     }
// }

// async function populateFeaturedAnimals() {
//     const featuredSection = document.querySelector('.featured-grid');
//     if (!featuredSection) return;
    
//     const animals = await fetchAnimalData();
//     const featuredAnimals = animals.filter(animal => animal.featured);
    
//     featuredAnimals.forEach(animal => {
//         const card = document.createElement('div');
//         card.className = 'animal-card';
//         card.innerHTML = `
//             <img src="${animal.image}" alt="${animal.name}">
//             <h3>${animal.displayName}</h3>
//             <p>${animal.category}</p>
//         `;
//         featuredSection.appendChild(card);
//     });
// }

// const currentPath = window.location.pathname;
// if (currentPath === '/' || currentPath.endsWith('index.html')) {
//     populateFeaturedAnimals();
// }
