//Blueket Template Dark Mode
const chk = document.getElementById('blueket');
chk.addEventListener('click', () => {
  chk.checked?document.body.classList.add("darkmode"):document.body.classList.remove("darkmode");
  localStorage.setItem('darkModeStatus', chk.checked);
});
window.addEventListener('load', (event) => {
  if(localStorage.getItem('darkModeStatus')=="true"){
    document.body.classList.add("darkmode"); 
    document.getElementById('blueket').checked = true;
  }
});
